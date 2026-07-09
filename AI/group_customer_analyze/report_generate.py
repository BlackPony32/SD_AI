import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple, Type

from AI.utils import get_logger, extract_customer_id, process_fetch_results, validate_save_results, generate_file_paths, create_response, \
    analyze_customer_orders_async, calculate_cost, is_data_ready, raw_filename_for
from AI.group_customer_analyze.create_report_group_c import create_agent_sectioned, create_agent_products_state_analysis
logger2 = get_logger("logger2", "project_log_many.log", False)
from agents import Agent, Runner
import aiofiles

from AI.group_customer_analyze.fetch_data import AnalysisIdType  # adjust import path if these live elsewhere
from AI.group_customer_analyze.create_report_group_c import (
    main_batch_process,
    generate_analytics_report_sectioned,
    combine_sections,
)
from AI.group_customer_analyze.orders_group import group_orders_statistics, main_batch_orders_process
from AI.group_customer_analyze.orders_state import async_generate_report, async_process_data
from enum import Enum
from AI.MCP_tools.additional_functions import (_calculate_key_metrics_orders, _calculate_discount_distribution, _calculate_orders_fulfillment, _calculate_payment_status, 
                                                   _calculate_sales_orders_performance, _sales_trends_orders_report)


import pandas as pd



class BaseReportGenerator(ABC):
    id_type: AnalysisIdType

    # Which ReportType values this generator knows how to produce.
    # None = no restriction. Subclasses that only support a subset
    # should set this explicitly, so an unsupported combo fails with a
    # clear message instead of a confusing downstream KeyError.
    supported_report_types: Optional[Set[Any]] = None

    def check_supported(self, report_type: Any) -> None:
        if self.supported_report_types is not None and report_type not in self.supported_report_types:
            raise ValueError(
                f"report_type={report_type.value!r} is not supported for "
                f"id_type={self.id_type.value!r}. Supported: "
                f"{sorted(rt.value for rt in self.supported_report_types)}"
            )

    @abstractmethod
    async def generate(
        self,
        report_type: Any,
        merged_orders,
        products_df,
        customer_df,
        uuid: str,
        start_time: float,
    ) -> Tuple[dict, Any]:
        """
        Returns (sections, full_report) on success. Raise on failure -
        the endpoint's except turns that into the 500 response.
        """
        ...


class CustomerReportGenerator(BaseReportGenerator):
    """
    Existing customer-analysis logic, lifted as-is from the endpoint.
    Everything here is specific to customer-shaped data - customer_df is
    meaningful, and AI.group_customer_analyze.* agents are built around
    customer-profile exports - which is why it isn't reused for orders.
    """
    id_type = AnalysisIdType.CUSTOMER
    supported_report_types = None  # full_report / product_per_state_analysis / any topic string

    async def generate(self, report_type, merged_orders, products_df, customer_df, uuid, start_time):
        self.check_supported(report_type)
        print(f"Step 4 - Before generate report (Type: {report_type.value}): {time.perf_counter() - start_time:.2f}s")

        if report_type.value == "full_report":
            return await self._full_report(merged_orders, products_df, customer_df, uuid, start_time)
        elif report_type.value == "product_per_state_analysis":
            return await self._product_per_state_analysis(merged_orders, products_df, customer_df, uuid, start_time)
        else:
            return await self._sectioned_topic(report_type, merged_orders, products_df, customer_df, uuid, start_time)

    async def _full_report(self, merged_orders, products_df, customer_df, uuid, start_time):
        try:
            full_report, sectioned_report = await main_batch_process(merged_orders, products_df, customer_df, uuid)

            async with aiofiles.open(f"data/{uuid}/full_report.txt", "w", encoding="utf-8") as f:
                await f.write(full_report)

            print("Step 5 - after generate report:", time.perf_counter() - start_time)
            return sectioned_report, full_report

        except Exception as e:
            logger2.warning(f"The problem of displaying of statistics in the 'full_report' block: {e}")
            try:
                raw_stats = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                return raw_stats.get('sections'), raw_stats.get('full_report')
            except Exception as e2:
                logger2.warning(f"The problem of displaying an alternative version of statistics in the 'full_report' block: {e2}")
                raise

    async def _product_per_state_analysis(self, merged_orders, products_df, customer_df, uuid, start_time):
        try:
            try:
                products_df['product_variant'] = products_df['name'].astype(str) + ' - ' + products_df['sku'].astype(str)
                await asyncio.gather(
                    asyncio.to_thread(merged_orders.to_csv, f'data/{uuid}/oorders.csv', index=False),
                    asyncio.to_thread(products_df.to_csv, f'data/{uuid}/pproducts.csv', index=False)
                )
            except Exception as e:
                logger2.warning(f"Error saving debug CSVs for {uuid}: {e}")

            await async_process_data(uuid)
            await async_generate_report(uuid)

            agent = await create_agent_products_state_analysis(uuid)

            answer = None
            try:
                runner = await Runner.run(agent, input="Based on the data return response")
                answer = runner.final_output
                for i in range(len(runner.raw_responses)):
                    print("Token usage : ", runner.raw_responses[i].usage, '')
            except Exception as e:
                print(f"Error in product_per_state_analysis runner: {e}")

            full_report = {}
            try:
                full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                    await f.write(full_report.get('full_report'))
                print("Step 5 - after generate report:", time.perf_counter() - start_time)
            except Exception as e:
                logger2.error(f"full report error in 'state' topic generate: {e}")

            sectioned_report = {'product_per_state_analysis': answer}
            return sectioned_report, full_report.get('full_report')

        except Exception as e:
            logger2.warning(f"The problem of displaying of statistics in the 'product_per_state_analysis' block: {e}")
            try:
                full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                sectioned_report = {
                    'product_per_state_analysis': (
                        'You do not have enough data to analyze the states, or '
                        'they do not meet the standards. Please try again later.'
                    )
                }
                return sectioned_report, full_report.get('full_report')
            except Exception as e2:
                logger2.warning(f"The problem of displaying an alternative version of statistics in the 'product_per_state_analysis' block: {e2}")
                raise

    async def _sectioned_topic(self, report_type, merged_orders, products_df, customer_df, uuid, start_time):
        topic = report_type.value
        try:
            statistics_of_topic = await generate_analytics_report_sectioned(
                merged_orders, products_df, customer_df, uuid, report_type=topic
            )
            agent = await create_agent_sectioned(uuid, topic, statistics_of_topic)

            runner = await Runner.run(agent, input="Based on the data return response")
            answer = runner.final_output
            sectioned_answer = await combine_sections(topic, statistics_of_topic, answer)

            calculate_cost(runner, model="gpt-5.4-mini")

            full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
            async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                await f.write(full_report.get('full_report'))

            print("Step 5 - after generate report:", time.perf_counter() - start_time)
            return sectioned_answer, full_report.get('full_report')

        except Exception as e:
            logger2.warning(f"The problem of displaying of statistics in the {topic} block: {e}")
            try:
                raw_stats = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid, topic)
                sectioned_report = {topic: raw_stats}
                return sectioned_report, raw_stats
            except Exception as e2:
                logger2.warning(f"The problem of displaying an alternative version of statistics in the {topic} block: {e2}")
                raise


class SalesReportGenerator(BaseReportGenerator):

    async def generate(self, report_type, cleaned_orders_path, cleaned_products_path, uuid, start_time):
        self.check_supported(report_type)
        print(f"Step 4 - Before generate report (Type: {report_type.value}): {time.perf_counter() - start_time:.2f}s")

        if report_type.value == "full_report":
            return await self._full_report(report_type, cleaned_orders_path, cleaned_products_path, uuid, start_time)
        else:
            return await self._sectioned_topic(report_type, cleaned_orders_path, cleaned_products_path, uuid, start_time)

    async def _full_report(self, report_type, cleaned_orders_path, cleaned_products_path, uuid, start_time):
        try:
            full_report, sections = await main_batch_orders_process(
                orders_path=cleaned_orders_path,
                products_path=cleaned_products_path,
                uuid=uuid,
                agent="orders_agent"
            )
            return sections, full_report

        except Exception as e:
            logger2.warning(f"The problem of displaying of statistics in the 'full_report' block: {e}")
            try:
                raw_stats = await group_orders_statistics(
                    orders_path=cleaned_orders_path,
                    products_path=cleaned_products_path,
                    agent_type="orders_agent",
                    report_type="full_report"
                )
                return raw_stats.get('sections'), raw_stats.get('full_report')
            except Exception as e2:
                logger2.warning(f"The problem of displaying an alternative version of statistics in the 'full_report' block: {e2}")
                raise

    async def _sectioned_topic(self, report_type, cleaned_orders_path, cleaned_products_path, uuid, start_time):
        topic = report_type.value
        try:
            statistics_of_topic = await group_orders_statistics(
                orders_path=cleaned_orders_path,
                products_path=cleaned_products_path,
                agent_type="orders_agent",
                report_type=topic
            )
            parsed_statistics = statistics_of_topic.get("sections", {}).get(report_type, "Report section not found.")

            agent = await create_agent_sectioned(uuid, topic, parsed_statistics)
            runner = await Runner.run(agent, input="Based on the data return response")
            answer = runner.final_output

            sectioned_answer = await combine_sections(topic, parsed_statistics, answer)

            calculate_cost(runner, model="gpt-5.4-mini")

            full_report = await group_orders_statistics(
                orders_path=cleaned_orders_path,
                products_path=cleaned_products_path,
                agent_type="orders_agent",
                report_type="full_report"
            )

            async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                await f.write(full_report.get('full_report'))

            print("Step 5 - after generate report:", time.perf_counter() - start_time)
            return sectioned_answer, full_report.get('full_report')

        except Exception as e:
            logger2.warning(f"The problem of displaying of statistics in the {topic} block: {e}")
            try:
                raw_stats = await group_orders_statistics(
                    orders_path=cleaned_orders_path,
                    products_path=cleaned_products_path,
                    agent_type="orders_agent",
                    report_type="full_report"
                )
                parsed_section = raw_stats.get("sections", {}).get(report_type, "Report section not found.")
                sectioned_report = {topic: parsed_section}
                full_report = raw_stats.get('full_report', "Full report not found.")

                return sectioned_report, full_report 
            except Exception as e2:
                logger2.warning(f"The problem of displaying an alternative version of statistics in the {topic} block: {e2}")
                raise


_REPORT_GENERATOR_REGISTRY: Dict[AnalysisIdType, Type[BaseReportGenerator]] = {
    AnalysisIdType.CUSTOMER: CustomerReportGenerator,
    AnalysisIdType.ORDER: SalesReportGenerator,
}


def get_report_generator(id_type: AnalysisIdType) -> BaseReportGenerator:
    try:
        generator_cls = _REPORT_GENERATOR_REGISTRY[id_type]
    except KeyError:
        raise ValueError(f"No report generator registered for id_type={id_type!r}")
    return generator_cls()