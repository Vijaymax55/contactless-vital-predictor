import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
import pandas as pd
import plotly.graph_objs as go
import urllib.parse
import datetime
from dash import dash_table
from conf.paths import data_path
import glob
import json  # Import the json module

root_folder = f"{data_path}"

available_dates = [
        date for date in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, date))
    ]

# Convert your initial date format (e.g., "20231018") to the normal format (e.g., "2023-10-18")
converted_dates = [datetime.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d") for date in available_dates if date.isdigit() and len(date) == 8]

start_date = min(converted_dates)
end_date = datetime.datetime.now()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("BluSim Data Dashboard", className="text-center mb-4"),

    html.Div([
        dcc.Store(id='button-state', storage_type='memory'),
        dbc.Button("One View", id='one-view-button', n_clicks=0, className="btn btn-lg"),
        html.Div(style={'width': '1cm'}),
        dbc.Button("Individual View", id='individual-view-button', n_clicks=0, className="btn btn-lg"),
        html.Div(style={'width': '1cm'}),
        dbc.Button("Accuracy", id='accuracy-button', n_clicks=0, className="btn btn-lg"),
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}, className="mt-4"),
    html.Div([
        dcc.DatePickerSingle(
            id='calendar',
            display_format='YYYY-MM-DD',
            max_date_allowed=end_date,
            min_date_allowed=start_date,
            initial_visible_month=end_date,
        )
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}, className="mt-4"),

    

    html.Div(id='modal', style={'display': 'none'}, children=[
        html.Div(id='modal-content', className='modal-content', children=[
            html.Span('Please choose a date before continuing.', className='close', id='close-modal'),
    ]),
    ]),    
    dcc.Loading(
        id="loading",
        type="default",
        fullscreen=True,
        children=[
            html.Div(id='view-content')
        ]
    )
])

@app.callback(
    [Output('one-view-button', 'style'), Output('individual-view-button', 'style'), Output('button-state', 'data')],
    [Input('one-view-button', 'n_clicks'), Input('individual-view-button', 'n_clicks'), Input('accuracy-button', 'n_clicks')],
    [dash.dependencies.State('button-state', 'data')],prevent_initial_call=True
)
def update_button_state(one_view_clicks, individual_view_clicks, accuracy_clicks, stored_data):
    if stored_data is None:
        stored_data = {'active_button': None}
    active_button = stored_data.get('active_button')
    if one_view_clicks is not None and (active_button != 'one-view-button' or active_button is None):
        stored_data['active_button'] = 'one-view-button'
        one_view_color = 'green'
        individual_view_color = 'blue'
    elif individual_view_clicks is not None and (active_button != 'individual-view-button' or active_button is None):
        stored_data['active_button'] = 'individual-view-button'
        one_view_color = 'blue'
        individual_view_color = 'green'
    elif accuracy_clicks is not None and (active_button != 'accuracy-button' or active_button is None):
        stored_data['active_button'] = 'accuracy-button'
        one_view_color = 'blue'
        individual_view_color = 'blue'
    else:
        if active_button == 'one-view-button':
            one_view_color = 'green'
            individual_view_color = 'blue'
        elif active_button == 'individual-view-button':
            one_view_color = 'blue'
            individual_view_color = 'green'
        else:
            one_view_color = 'blue'
            individual_view_color = 'blue'
    one_view_style = {'backgroundColor': one_view_color}
    individual_view_style = {'backgroundColor': individual_view_color}
    return one_view_style, individual_view_style, stored_data

@app.callback(
    [Output('modal', 'style'), Output('view-content', 'children')],
    [Input('one-view-button', 'n_clicks'), Input('individual-view-button', 'n_clicks'), Input('accuracy-button', 'n_clicks'), Input('calendar', 'date')],
    [dash.dependencies.State('button-state', 'data')],
    prevent_initial_call=True
)
def switch_view(one_view_clicks, individual_view_clicks, accuracy_clicks, selected_date, stored_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'none'}, None

    with open("conf/variables.json", "r") as j_file:
        loaded_data = json.load(j_file)

    nrm_date = selected_date.replace("-", "") if selected_date else None
    loaded_data["selected_date"] = nrm_date
    with open("conf/variables.json", "w") as file:
        json.dump(loaded_data, file, indent=4)

    if ctx.triggered[0]['prop_id'] == 'one-view-button.n_clicks':
        if not selected_date:
            return {'display': 'block'}, None
        else:
            return {'display': 'none'}, one_view_layout(nrm_date, selected_date)
    elif ctx.triggered[0]['prop_id'] == 'individual-view-button.n_clicks':
        if not selected_date:
            return {'display': 'block'}, None
        else:
            return {'display': 'none'}, individual_view_layout(nrm_date)
    elif ctx.triggered[0]['prop_id'] == 'accuracy-button.n_clicks':
        return {'display': 'none'}, accuracy_layout(nrm_date)
    else:
        return {'display': 'none'}, None  


#for merging all the files present in the specified folder by looking at folder name
def merge_csv_files_in_directory(directory):
    # Get a list of CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    # Sort the CSV files based on their Unix timestamps in the file names
    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Read and merge the CSV files
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        merged_df = pd.concat([merged_df, data])

    return merged_df
import execution
def one_view_layout(selected_date, nrm_date):
    execute =  execution.exe(nrm_date)
    if execute:
        if "-" in selected_date:
            selected_date = selected_date.replace("-", "")

        date_obj = datetime.datetime.strptime(selected_date, "%Y%m%d")
        # Format the date object as "YYYY-MM-DD"
        formatted_date = date_obj.strftime("%Y-%m-%d")
        '''print(formatted_date)
        print(nrm_date)'''
        sleep_cycle_data = pd.read_csv(fr'{root_folder}\{selected_date}\sleep_cycle.csv')
        mat_data = merge_csv_files_in_directory(fr'{root_folder}\{selected_date}\mat_csv')
        #para_monitor_data = pd.read_csv(fr'{root_folder}\{selected_date}\para_csv\para_monitor_{nrm_date}.csv')
        sleep_pose_data = pd.read_csv(fr'{root_folder}\{selected_date}\sleep_pose.csv')

        '''desired_columns = [
            'Time', 'label', 'Chn_1', 'Chn_2', 'Chn_3', 'Chn_4',
            'RESP', 'Sistolic','Distolic', 'ECG', 'SPO2', 'Pose'
        ]'''
        desired_columns = ['Time', 'label', 'Chn_1', 'Chn_2', 'Chn_3', 'Chn_4','Pose']

        merged_df1 = pd.merge( mat_data,sleep_cycle_data, on='Time', how='inner')
        #merged_df1 = pd.merge(merged_df1, para_monitor_data, on='Time', how='inner')
        merged_df1 = pd.merge(merged_df1, sleep_pose_data, on='Time', how='inner')

        merged_df1 = merged_df1[desired_columns]
        table_data = merged_df1.to_dict('records')
        csv_string = merged_df1.to_csv(index=False, encoding='utf-8')
        csv_data = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

        download_button = html.Div(
            html.A(
                "Download CSV",
                href=csv_data,
                download="combined_data.css",
                style={
                    'background-color': '#007BFF',
                    'color': 'white',
                    'border': '1px solid #0056b3',
                    'padding': '10px 20px',
                    'margin': '20px',
                    'text-decoration': 'none',
                    'border-radius': '5px',
                    'display': 'inline-block',
                    'cursor': 'pointer'
                }
            ),
            style={'text-align': 'center'}
        )

        table = dash_table.DataTable(
            id='data-table-one-view',
            data=table_data,
            style_table={'overflowX': 'scroll'},
        )

        container = html.Div([
            download_button,
            table,
        ], style={'margin': '20px'})

        view_layout = html.Div([
            container,
        ])

        return view_layout
    return dcc.Loading(
        type="default",
        fullscreen=True,
        children=[
            html.Div("Loading...")  # Display a loading message
        ]
    )



def individual_view_layout(selected_date):
    file_path = 'conf/share.txt'
    with open(file_path, "w") as file:
        file.write(selected_date)
    if "-" in selected_date:
        selected_date = selected_date.replace("-", "")
    date_obj = datetime.datetime.strptime(selected_date, "%Y%m%d")
    # Format the date object as "YYYY-MM-DD"
    sleep_cycle_data = pd.read_csv(fr'{root_folder}\{selected_date}\sleep_cycle.csv')
    mat_data = merge_csv_files_in_directory(fr'{root_folder}\{selected_date}\mat_csv')
    #para_monitor_data = pd.read_csv(fr'{root_folder}\{selected_date}\para_csv\para_monitor_{nrm_date}.csv')
    sleep_pose_data = pd.read_csv(fr'{root_folder}\{selected_date}\sleep_pose.csv')

    dropdown = dcc.Dropdown(
        id='graph-selector',
        options=[
            {'label': 'Sleep Cycle', 'value': 'sleep_cycle'},
            {'label': 'Mat Data', 'value': 'mat_data'},
            #{'label': 'Para Monitor', 'value': 'para_monitor'},
            {'label': 'Sleep Pose', 'value': 'sleep_pose'},
        ],
        value='sleep_cycle',
    )
    graph = dcc.Graph(id='data-graph')
    table = dash_table.DataTable(id='data-table-individual-view', style_table={'overflowX': 'scroll'})

    return html.Div([dropdown, graph, table])

def accuracy_layout(date):
    if "-" in date:
        date = date.replace("-", "")
    accuracy_file_path = pd.read_csv(rf'{root_folder}\{date}\accuracy_prediction\accuracy.csv')
    table_data = accuracy_file_path.to_dict('records')
    table = dash_table.DataTable(
        id='data-table-one-view',
        data=table_data,
        style_table={'overflowX': 'scroll'},
    )

    container = html.Div([
        table,
    ], style={'margin': '20px'})

    view_layout = html.Div([
        container,
    ])

    return view_layout

@app.callback(
    [Output('data-graph', 'figure'), Output('data-table-individual-view', 'data')],
    [Input('graph-selector', 'value'), Input('calendar', 'date')],prevent_initial_call=True
)
def update_graph(selected_graph, selected_date):
    
    if "-" in selected_date:
        selected_date = selected_date.replace("-", "")
    traces = []
    date_obj = datetime.datetime.strptime(selected_date, "%Y%m%d")
    nrm_date = date_obj.strftime("%Y-%m-%d")
    print(selected_date)
    print(nrm_date)
    if selected_graph == 'sleep_cycle':
        sleep_cycle_data = pd.read_csv(f'{root_folder}/{selected_date}/sleep_cycle.csv')
        trace = go.Scatter(x=sleep_cycle_data['Time'], y=sleep_cycle_data['label'], mode='lines+markers')
        layout = go.Layout(title='Sleep Cycle', xaxis={'title': 'Time'}, yaxis={'title': 'Label'})
        traces = [trace]
        table_data = sleep_cycle_data.to_dict('records')
    elif selected_graph == 'mat_data':
        mat_data = merge_csv_files_in_directory(f'{root_folder}/{selected_date}/mat_csv')
        for channel in ['Chn_1', 'Chn_2', 'Chn_3', 'Chn_4']:
            trace = go.Scatter(x=mat_data['Time'], y=mat_data[channel], mode='lines', name=channel)
            traces.append(trace)
        layout = go.Layout(title='Mat Data', xaxis={'title': 'Time'}, yaxis={'title': 'Value'})
        table_data = mat_data.to_dict('records')
    #elif selected_graph == 'para_monitor':
    #    para_monitor_data = pd.read_csv(f'{root_folder}/{selected_date}/para_csv/para_monitor_{nrm_date}.csv')
    #    for parameter in ['RESP', 'NIBP', 'ECG', 'SPO2', 'PR']:
    #        trace = go.Scatter(x=para_monitor_data['Timestamp'], y=para_monitor_data[parameter], mode='lines', name=parameter)
    #        traces.append(trace)
    #   layout = go.Layout(title='Para Monitor Data', xaxis={'title': 'Timestamp'}, yaxis={'title': 'Value'})
    #    table_data = para_monitor_data.to_dict('records')
    elif selected_graph == 'sleep_pose':
        sleep_pose_data = pd.read_csv(f'{root_folder}/{selected_date}/sleep_pose.csv')
        trace = go.Scatter(x=sleep_pose_data['Time'], y=sleep_pose_data['Pose'], mode='lines')
        layout = go.Layout(title='Sleep Pose', xaxis={'title': 'Time'}, yaxis={'title': 'Pose'})
        traces.append(trace)

        table_data = sleep_pose_data.to_dict('records')
    else:
        table_data = []

    return {'data': traces, 'layout': layout}, table_data

if __name__ == "__main__":
    app.run_server(debug=True)
