import gradio as gr
import requests
import pandas as pd
import markdown
from functools import partial
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()
import logging
log_file_path = "project_log.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('project_log.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
def generate_insights_plot(customer_id, selected_metrics):
    # Dynamic file paths using customer_id
    products_path = f'data/{customer_id}/work_prod.csv'
    orders_path = f'data/{customer_id}/work_ord.csv'
    
    products_df = pd.read_csv(products_path)
    products_df = products_df.drop(columns=['id'])
    products_df['createdAt'] = pd.to_datetime(products_df['createdAt'], utc=True)

    orders_df = pd.read_csv(orders_path)
    orders_df = orders_df.rename(columns={'id':'orderId'})
    orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'], utc=True)

    merged_df = pd.merge(products_df, orders_df, on=['orderId', 'createdAt', 'totalAmount'], how='left')

    daily_sales = merged_df.groupby(merged_df['createdAt'].dt.date).agg({
        'totalAmount': 'sum',
        'quantity': 'sum'
    }).reset_index()

    status_counts = orders_df.groupby([orders_df['createdAt'].dt.date, 'orderStatus']).size().unstack(fill_value=0).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Conditional trace adding based on selected metrics
    if 'Total Quantity' in selected_metrics:
        fig.add_trace(
            go.Bar(
                x=daily_sales['createdAt'],
                y=daily_sales['quantity'],
                name='Total Quantity Sold',
                opacity=0.6,
                marker_color='red',
                hovertemplate="<b>Total Quantity Sold: </b> %{y:.2f}<extra></extra>"
            ),
            secondary_y=False,
        )
    
    if 'Total Sales' in selected_metrics:
        fig.add_trace(
            go.Scatter(
                x=daily_sales['createdAt'],
                y=daily_sales['totalAmount'],
                name='Total Sales',
                mode='lines+markers',
                line=dict(color='darkblue', width=2),
                hovertemplate="<b>Total Sales:</b> %{y:.2f}$<extra></extra>"
            ),
            secondary_y=False,
        )

    if 'Order Status' in selected_metrics:
        for status in status_counts.columns[1:]:
            fig.add_trace(
                go.Scatter(
                    x=status_counts['createdAt'],
                    y=status_counts[status],
                    name=f'{status} Orders',
                    mode='markers+lines',
                    marker=dict(size=10),
                    text=status,
                    hovertemplate= status + ": %{y:.2f}<extra></extra>"
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title='Business Insights: Sales, Quantity, and Order Status Over Time',
        xaxis_title='Date',
        yaxis_title='Sales ($) / Quantity',
        yaxis2_title='Number of Orders',
        legend_title='Metrics',
        barmode='group',
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(tickprefix='$', showgrid=True))
    
    return fig




def plot_sales_treemap(customer_id):
    try:
        products_path = f'data/{customer_id}/work_prod.csv'
        df = pd.read_csv(products_path)
        
        sku_level = df.groupby(['manufacturerName', 'sku']).agg(
            total_amount=('totalAmount', 'sum'),
            total_quantity=('quantity', 'sum')
        ).reset_index()
        
        manufacturer_level = df.groupby('manufacturerName').agg(
            manufacturer_total=('totalAmount', 'sum'),
            manufacturer_quantity=('quantity', 'sum')
        ).reset_index()

        top_label = 'All Manufacturers'
        labels = [top_label] + manufacturer_level['manufacturerName'].tolist() + sku_level['sku'].tolist()
        parents = [''] + [top_label] * len(manufacturer_level) + sku_level['manufacturerName'].tolist()
        values = [manufacturer_level['manufacturer_total'].sum()] + manufacturer_level['manufacturer_total'].tolist() + sku_level['total_amount'].tolist()
        quantities = [manufacturer_level['manufacturer_quantity'].sum()] + manufacturer_level['manufacturer_quantity'].tolist() + sku_level['total_quantity'].tolist()

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colors=quantities,
                colorscale='Oxy',
                showscale=True,
                cmin=min(quantities),
                cmax=max(quantities),
                colorbar_title='Total Quantity Sold'
            ),
            hovertemplate='<b>%{label}</b><br>Total Sales: $%{value:.2f}<br>Quantity Sold: %{customdata:.0f}<extra></extra>',
            customdata=quantities
        ))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, x=0.5, y=0.5)
        return fig

API_BASE_URL = os.getenv('FAST_API_UPL')

green_shades = gr.themes.Color(
    c50="#e6f4ea",
    c100="#c9e9d1",
    c200="#a3d9b1",
    c300="#7dc992",
    c400="#66bb6a",
    c500="#409A65",  # Your exact color
    c600="#358759",
    c700="#2a734b",
    c800="#1f5c3d",
    c900="#14462f",
    c950="#0a2d1f"
)
def generate_reports(customer_id: str):
    try:
        # Query parameters
        params = {
            "customer_id": customer_id,
            "entity": 'orders'
        }
        logger.info(f'API_BASE_URL: {API_BASE_URL}')
        #API_BASE_URL = 'http://sd_ai-fastapi-app-1:8000'
        response = requests.post(f"{API_BASE_URL}/generate-reports/{customer_id}", params=params)

        
        logger.info(f'response: {response.status_code}')
        if response.ok:
            data = response.json()
            return (
                data.get('file_path_product', ''),
                data.get('file_path_orders', ''),
                f"## Report Generated\n{data.get('report', '')}",
                True  # Success flag
            )
        else:
            logger.error(f"Error: {response.status_code}, Response: {response.text}")
        return '', '', f"Error: {response.text}", False
    except Exception as e:
        logger.error(e)
        return '', '', "Cannot answer this question", False

def ask_ai(message, file_paths, customer_id):
    product_path, orders_path = file_paths
    try:
        url = f"{API_BASE_URL}/Ask_ai"
        params = {
            "file_path_product": product_path,
            "file_path_orders": orders_path,
            "customer_id": customer_id
        }
        data = {
            "prompt": message
        }
        response = requests.post(url, params=params, json=data)
        if response.ok:
            data = response.json()
            output = data.get('response', {}).get('output', '')
            return markdown.markdown(str(output))
        return f"API Error: {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"

def create_interface():
    custom_theme = gr.themes.Default(primary_hue=green_shades)
    
    with gr.Blocks(title="API Tester", theme=custom_theme) as demo:
        file_paths = gr.State(('', ''))
        
        gr.Markdown("# API Test Interface")
        
        with gr.Row():
            customer_id = gr.Textbox(label="Customer ID", interactive=True)
            report_btn = gr.Button("Generate Reports", variant="primary")
        
        status = gr.Markdown()
        metrics_filter = gr.CheckboxGroup(
            label="Select Metrics to Display",
            choices=["Total Quantity", "Total Sales", "Order Status"],
            value=["Total Quantity", "Total Sales"],
            visible=False
        )
        # Add plot components that will be updated
        plot1 = gr.Plot()
        plot2 = gr.Plot()
        
        chatbot = gr.Chatbot(visible=False, type="messages", min_width=80, max_height=500)
        suggested_questions = [
            "What are the top-selling products?",
            "What is the total revenue?",
            "Top reorder periods?",
            "Top customer visit time?"
        ]
        
        with gr.Row():
            suggest_buttons = [gr.Button(q, visible=False) for q in suggested_questions]
        
        msg = gr.Textbox(label="Your Question", visible=False)
        submit_btn = gr.Button("Send", variant="primary", visible=False)
        
        def handle_report(customer_id):
            p_path, o_path, report, success = generate_reports(customer_id)
            if success:
                initial_chat = [{"role": "assistant", "content": report}]
                return (
                    "Reports generated. You can now ask questions below.",
                    gr.update(value=initial_chat, visible=True),
                    *[gr.update(visible=True) for _ in suggest_buttons],
                    gr.update(visible=True),
                    gr.update(visible=True),
                    (p_path, o_path),
                    generate_insights_plot(customer_id, ["Total Quantity", "Total Sales"]),
                    plot_sales_treemap(customer_id),
                    gr.update(visible=True)  # For metrics filter
                )
            else:
                # Create empty figure instead of trying to read files
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="No plots available", showarrow=False, x=0.5, y=0.5)
                return (
                    report,
                    gr.update(visible=False),
                    *[gr.update(visible=False) for _ in suggest_buttons],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    ('', ''),
                    empty_fig,  # Return empty figure instead of generating from invalid path
                    empty_fig,
                    gr.update(visible=False)
                )
        
        report_btn.click(
            handle_report,
            inputs=[customer_id],
            outputs=[
                status, chatbot, *suggest_buttons, msg, submit_btn, 
                file_paths, plot1, plot2, metrics_filter
            ]
        )
        
        # Add metrics filter update handler
        metrics_filter.change(
            fn=generate_insights_plot,
            inputs=[customer_id, metrics_filter],
            outputs=plot1
        )
        
        def handle_chat(message, chat_history, paths, customer_id):
            response = ask_ai(message, paths, customer_id)
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history
        
        submit_btn.click(
            handle_chat,
            inputs=[msg, chatbot, file_paths, customer_id],
            outputs=[msg, chatbot]
        )

        msg.submit(
            handle_chat,
            inputs=[msg, chatbot, file_paths, customer_id],
            outputs=[msg, chatbot]
        )
        
        for btn, question in zip(suggest_buttons, suggested_questions):
            btn.click(
                partial(handle_chat, question),
                inputs=[chatbot, file_paths, customer_id],
                outputs=[msg, chatbot]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=7860,
    share=True
)
