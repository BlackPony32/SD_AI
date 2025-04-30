import gradio as gr
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial
import markdown
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

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




# Define custom green theme
green_shades = gr.themes.Color(
    c50="#e6f4ea",
    c100="#c9e9d1",
    c200="#a3d9b1",
    c300="#7dc992",
    c400="#66bb6a",
    c500="#409A65",
    c600="#358759",
    c700="#2a734b",
    c800="#1f5c3d",
    c900="#14462f",
    c950="#0a2d1f"
)
custom_theme = gr.themes.Default(primary_hue=green_shades)


def generate_reports(customer_id: str, entity: str):
    try:
        params = {"customer_id": customer_id, "entity": entity}
        API_BASE_URL = os.getenv('FAST_API_UPL')
        response = requests.post(f"{API_BASE_URL}/generate-reports/{customer_id}", params=params)
        if response.ok:
            data = response.json()
            return (
                data.get('file_path_product', ''),
                data.get('file_path_orders', ''),
                f"## Report Generated for {entity}\n{data.get('report', '')}",
                True
            )
        else:
            return '', '', f"Error: {response.text}", False
    except Exception as e:
        return '', '', "Cannot answer this question", False

def ask_ai(message, file_paths, customer_id):
    product_path, orders_path = file_paths
    try:
        API_BASE_URL = os.getenv('FAST_API_UPL')
        url = f"{API_BASE_URL}/Ask_ai"
        params = {"file_path_product": product_path, "file_path_orders": orders_path, "customer_id": customer_id}
        data = {"prompt": message}
        response = requests.post(url, params=params, json=data)
        if response.ok:
            data = response.json()
            output = data.get('response', {}).get('output', '')
            return markdown.markdown(str(output))
        return f"API Error: {response.text}"
    except Exception as e:
        return f"Can not answer on your question("

def handle_chat(message, chat_history, paths, customer_id):
    response = ask_ai(message, paths, customer_id)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history

def create_interface():
    with gr.Blocks(theme=custom_theme) as demo:
        file_paths = gr.State(('', ''))
        
        customer_id = gr.Textbox(label="Customer ID", interactive=True)
        show_plots_checkbox = gr.Checkbox(label="Show plots for orders report", value=True)
        
        with gr.Row():
            orders_btn = gr.Button("Generate Orders Report", variant="primary")
            activities_btn = gr.Button("Generate Activities Report", variant="primary")
        
        status = gr.Markdown()
        chatbot = gr.Chatbot(visible=False, type="messages")
        
        with gr.Row():
            suggest_buttons = [gr.Button(q, visible=False, variant="primary") for q in [
                "What are the top-selling products?",
                "What is the total revenue?",
                "Top reorder periods?",
                "Top customer visit time?"
            ]]
        
        msg = gr.Textbox(label="Your Question", visible=False)
        submit_btn = gr.Button("Send", variant="primary", visible=False)
        
        metrics_filter = gr.CheckboxGroup(
            label="Select Metrics to Display",
            choices=["Total Quantity", "Total Sales", "Order Status"],
            value=["Total Quantity", "Total Sales"],
            visible=False
        )
        plot1 = gr.Plot()
        plot2 = gr.Plot()
        
        def handle_orders_report(customer_id, show_plots):
            p_path, o_path, report, success = generate_reports(customer_id, 'orders')
            if success:
                initial_chat = [{"role": "assistant", "content": report}]
                if show_plots:
                    fig1 = generate_insights_plot(customer_id, ["Total Quantity", "Total Sales"])
                    fig2 = plot_sales_treemap(customer_id)
                    plot1_update = gr.update(value=fig1, visible=True)
                    plot2_update = gr.update(value=fig2, visible=True)
                    metrics_update = gr.update(visible=True)
                else:
                    plot1_update = gr.update(visible=False)
                    plot2_update = gr.update(visible=False)
                    metrics_update = gr.update(visible=False)
                return (
                    "Orders report generated. You can now ask questions below.",
                    gr.update(value=initial_chat, visible=True),
                    *[gr.update(visible=True) for _ in suggest_buttons],
                    gr.update(visible=True),
                    gr.update(visible=True),
                    (p_path, o_path),
                    plot1_update,
                    plot2_update,
                    metrics_update
                )
            else:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="No plots available", showarrow=False, x=0.5, y=0.5)
                return (
                    "Can not create report due to incorrect customer id",
                    gr.update(visible=False),
                    *[gr.update(visible=False) for _ in suggest_buttons],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    ('', ''),
                    gr.update(value=empty_fig, visible=True),
                    gr.update(value=empty_fig, visible=True),
                    gr.update(visible=False)
                )
        
        def handle_activities_report(customer_id):
            p_path, o_path, report, success = generate_reports(customer_id, 'activities')
            if success:
                initial_chat = [{"role": "assistant", "content": report}]
                return (
                    "Activities report generated.",
                    gr.update(value=initial_chat, visible=True),
                    *[gr.update(visible=False) for _ in suggest_buttons],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    (p_path, o_path),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            else:
                return (
                    "Can not create report due to incorrect customer id",
                    gr.update(visible=False),
                    *[gr.update(visible=False) for _ in suggest_buttons],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    ('', ''),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        orders_btn.click(
            handle_orders_report,
            inputs=[customer_id, show_plots_checkbox],
            outputs=[status, chatbot, *suggest_buttons, msg, submit_btn, file_paths, plot1, plot2, metrics_filter]
        )
        
        activities_btn.click(
            handle_activities_report,
            inputs=[customer_id],
            outputs=[status, chatbot, *suggest_buttons, msg, submit_btn, file_paths, plot1, plot2, metrics_filter]
        )
        
        metrics_filter.change(
            fn=generate_insights_plot,
            inputs=[customer_id, metrics_filter],
            outputs=plot1
        )
        
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
        
        for btn, question in zip(suggest_buttons, [
            "What are the top-selling products?",
            "What is the total revenue?",
            "Top reorder periods?",
            "Top customer visit time?"
        ]):
            btn.click(
                partial(handle_chat, question),
                inputs=[chatbot, file_paths, customer_id],
                outputs=[msg, chatbot]
            )
    
    return demo

#if __name__ == "__main__":
#    demo = create_interface()
#    demo.launch()
    
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=7860,
    share=True
)
