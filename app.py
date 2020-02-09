import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
# from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
# import data_analysis as da
# from settings import months, metadataDB
# import db_engine as db
# from db_info import db_info
import urllib.parse
import plotly.graph_objs as go
import re
import datetime
import base64
import io

app = dash.Dash(__name__)
server = app.server

metadataDB = pd.read_csv("MetadataDB.csv")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
USEPA_LIMIT = 4
WHO_LIMIT = 20

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
# initial data frame
empty_df = pd.DataFrame()

df1 = pd.read_csv("https://raw.githubusercontent.com/divyachandran-ds/dash1/master/Energy2.csv")
df = df1.dropna()


class db_info:
    def __init__(self, db_name, uploaded_by, institution):
        current_date = datetime.datetime.now()
        self.db_name = db_name
        self.uploaded_by = uploaded_by
        self.institution = institution
        self.upload_date = current_date.strftime("%Y\%m\%d")
        self.db_id = db_name.replace(" ", "_") + '_' + uploaded_by.replace(" ", "_") + '_' + current_date.strftime(
            "%Y\%m\%d")

        self.db_publication_url = ''
        self.db_field_method_url = ''
        self.db_lab_method_url = ''
        self.db_QAQC_url = ''
        self.db_full_QCQC_url = ''
        self.db_substrate = ''
        self.db_sample_type = ''
        self.db_field_method = ''
        self.db_microcystin_method = ''
        self.db_filter_size = ''
        self.db_cell_count_method = ''
        self.db_ancillary_url = ''
        self.db_num_lakes = 0
        self.db_num_samples = 0


def convert_to_json(current_dataframe):
    '''
        converts all the data to a JSON string
    '''
    jsonStr = current_dataframe.to_json(orient='split')
    return jsonStr


def convert_to_df(jsonified_data):
    '''
        converts the JSON string back to a dataframe
    '''
    jsonStr = r'{}'.format(jsonified_data)
    dff = pd.read_json(jsonStr, orient='split')
    return dff


def get_metadata_table_content(current_metadata):
    '''
        returns the data for the specified columns of the metadata data table
    '''

    table_df = current_metadata[
        ['DB_ID', 'DB_name', 'Uploaded_by', 'Upload_date', 'Microcystin_method', 'N_lakes', 'N_samples']]
    return table_df.to_dict("rows")


def upload_new_database(new_dbinfo, contents, filename):
    """
        Decode contents of the upload component and create a new dataframe
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            new_df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return parse_new_database(new_dbinfo, new_df)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            new_df = pd.read_excel(io.BytesIO(decoded))
            return parse_new_database(new_dbinfo, new_df)
        else:
            return 'Invalid file type.'
    except Exception as e:
        print(e)
        return 'There was an error processing this file.'


def parse_new_database(new_dbinfo, new_df):
    """
        Convert CSV or Excel file data into Pickle file and store in the data directory
    """
    try:

        # delete the extra composite section of the lake names - if they have any
        new_df['LakeName'] = new_df['LakeName']. \
            str.replace(r"[-]?.COMPOSITE(.*)", "", regex=True). \
            str.strip()

        new_df['Date'] = pd.to_datetime(new_df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        print(new_df['LakeName'])
        # convert mg to ug
        new_df['TP_mgL'] *= 1000
        new_df['TN_mgL'] *= 1000

        # format all column names
        new_df.rename(columns={'Date': 'DATETIME',
                               'LakeName': 'Body of Water Name',
                               'Lat': 'LAT',
                               'Long': 'LONG',
                               'Altitude_m': 'Altitude (m)',
                               'MaximumDepth_m': 'Maximum Depth (m)',
                               'MeanDepth_m': 'Mean Depth (m)',
                               'SecchiDepth_m': 'Secchi Depth (m)',
                               'SamplingDepth_m': 'Sampling Depth (m)',
                               'ThermoclineDepth_m': 'Thermocline Depth (m)',
                               'SurfaceTemperature_C': 'Surface Temperature (degrees celsius)',
                               'EpilimneticTemperature_C': 'Epilimnetic Temperature (degrees celsius)',
                               'TP_mgL': 'Total Phosphorus (ug/L)',
                               'TN_mgL': 'Total Nitrogen (ug/L)',
                               'NO3NO2_mgL': 'NO3 NO2 (mg/L)',
                               'NH4_mgL': 'NH4 (mg/L)',
                               'PO4_ugL': 'PO4 (ug/L)',
                               'Chlorophylla_ugL': 'Total Chlorophyll a (ug/L)',
                               'Chlorophyllb_ugL': 'Total Chlorophyll b (ug/L)',
                               'Zeaxanthin_ugL': 'Zeaxanthin (ug/L)',
                               'Diadinoxanthin_ugL': 'Diadinoxanthin (ug/L)',
                               'Fucoxanthin_ugL': 'Fucoxanthin (ug/L)',
                               'Diatoxanthin_ugL': 'Diatoxanthin (ug/L)',
                               'Alloxanthin_ugL': 'Alloxanthin (ug/L)',
                               'Peridinin_ugL': 'Peridinin (ug/L)',
                               'Chlorophyllc2_ugL': 'Total Chlorophyll c2 (ug/L)',
                               'Echinenone_ugL': 'Echinenone (ug/L)',
                               'Lutein_ugL': 'Lutein (ug/L)',
                               'Violaxanthin_ugL': 'Violaxanthin (ug/L)',
                               'TotalMC_ug/L': 'Microcystin (ug/L)',
                               'DissolvedMC_ugL': 'DissolvedMC (ug/L)',
                               'MC_YR_ugL': 'Microcystin YR (ug/L)',
                               'MC_dmRR_ugL': 'Microcystin dmRR (ug/L)',
                               'MC_RR_ugL': 'Microcystin RR (ug/L)',
                               'MC_dmLR_ugL': 'Microcystin dmLR (ug/L)',
                               'MC_LR_ugL': 'Microcystin LR (ug/L)',
                               'MC_LY_ugL': 'Microcystin LY (ug/L)',
                               'MC_LW_ugL': 'Microcystin LW (ug/L)',
                               'MC_LF_ugL': 'Microcystin LF (ug/L)',
                               'NOD_ugL': 'Nodularin (ug/L)',
                               'CYN_ugL': 'Cytotoxin Cylindrospermopsin (ug/L)',
                               'ATX_ugL': 'Neurotoxin Anatoxin-a (ug/L)',
                               'GEO_ugL': 'Geosmin (ug/L)',
                               '2MIB_ngL': '2-MIB (ng/L)',
                               'TotalPhyto_CellsmL': 'Phytoplankton (Cells/mL)',
                               'Cyano_CellsmL': 'Cyanobacteria (Cells/mL)',
                               'PercentCyano': 'Relative Cyanobacterial Abundance (percent)',
                               'DominantBloomGenera': 'Dominant Bloom',
                               'mcyD_genemL': 'mcyD gene (gene/mL)',
                               'mcyE_genemL': 'mcyE gene (gene/mL)', },
                      inplace=True)

        # remove NaN columns
        new_df = new_df.dropna(axis=1, how='all')

        # save the pickle and csv file in the data directory
        pkldir = get_pkl_path(new_dbinfo.db_id)
        new_df.to_pickle(pkldir)

        csvdir = get_csv_path(new_dbinfo.db_id)
        new_df.to_csv(csvdir)

        # update the number of lakes and samples in db_info
        unique_lakes_list = list(new_df["Body of Water Name"].unique())
        new_dbinfo.db_num_lakes = len(unique_lakes_list)
        new_dbinfo.db_num_samples = new_df.shape[0]

        current_metadata = metadataDB
        update_metadata(new_dbinfo, current_metadata)
        return u'''Database "{}" has been successfully uploaded.'''.format(new_dbinfo.db_name)

    except Exception as e:
        print(e)
        return 'Error uploading database'


def update_metadata(new_dbinfo, current_metadata):
    """
        Add new database info to MetadataDB.csv
    """
    try:
        current_metadata = pd.read_csv("MetadataDB.csv")

        new_dbdf = pd.DataFrame({'DB_ID': [new_dbinfo.db_id],
                                 'DB_name': [new_dbinfo.db_name],
                                 'Uploaded_by': [new_dbinfo.uploaded_by],
                                 'Upload_date': [new_dbinfo.upload_date],
                                 'Published_url': [new_dbinfo.db_publication_url],  # url
                                 'Field_method_url': [new_dbinfo.db_field_method_url],  # url
                                 'Lab_method_url': [new_dbinfo.db_lab_method_url],  # url
                                 'QA_QC_url': [new_dbinfo.db_QAQC_url],  # url
                                 'Full_QA_QC_url': [new_dbinfo.db_full_QCQC_url],  # url
                                 'Substrate': [new_dbinfo.db_substrate],
                                 'Sample_type': [new_dbinfo.db_sample_type],
                                 'Field-method': [new_dbinfo.db_field_method],
                                 'Microcystin_method': [new_dbinfo.db_microcystin_method],
                                 'Filter_size': [new_dbinfo.db_filter_size],
                                 'Cell_count_method': [new_dbinfo.db_cell_count_method],
                                 'Ancillary_data': [new_dbinfo.db_ancillary_url],
                                 'N_lakes': [new_dbinfo.db_num_lakes],
                                 'N_samples': [new_dbinfo.db_num_samples]})

        metadataDB = pd.concat([current_metadata, new_dbdf], sort=False).reset_index(drop=True)
        metadataDB.to_csv("MetadataDB.csv", encoding='utf-8', index=False)
    except Exception as e:
        print(e)
        return 'Error saving metadata'


def update_dataframe(selected_rows):
    """
        update dataframe based on selected databases
    """
    try:
        new_dataframe = pd.DataFrame()
        # Read in data from selected Pickle files into Pandas dataframes, and concatenate the data
        for row in selected_rows:
            rowid = row["DB_ID"]
            filepath = get_pkl_path(rowid)
            db_data = pd.read_pickle(filepath)
            new_dataframe = pd.concat([new_dataframe, db_data], sort=False).reset_index(drop=True)

        # Ratio of Total Nitrogen to Total Phosphorus
        # This line causes a problem on certain datasets as the columns are strings instead of ints and will not divide, dataset dependent
        print(new_dataframe["Total Nitrogen (ug/L)"])
        print("Phosphorus: ", new_dataframe["Total Phosphorus (ug/L)"])
        new_dataframe["TN:TP"] = new_dataframe["Total Nitrogen (ug/L)"] / new_dataframe["Total Phosphorus (ug/L)"]
        # Ration of Microcystin to Total Chlorophyll
        new_dataframe["Microcystin:Chlorophyll a"] = new_dataframe["Microcystin (ug/L)"] / new_dataframe[
            "Total Chlorophyll a (ug/L)"]
        # Percent change of microcystin
        new_dataframe["MC Percent Change"] = new_dataframe.sort_values("DATETIME"). \
            groupby(['LONG', 'LAT'])["Microcystin (ug/L)"]. \
            apply(lambda x: x.pct_change()).fillna(0)
        return new_dataframe
    except Exception as e:
        print("EXCEPTION: ", e)


def get_pkl_path(db_id):
    return db_id + '.pkl'


def get_csv_path(db_id):
    return db_id + '.csv'


def geo_log_plot(selected_data, current_df):
    selected_data["MC_pc_bin"] = np.log(np.abs(selected_data["MC Percent Change"]) + 1)
    data = [go.Scattergeo(
        lon=selected_data['LONG'],
        lat=selected_data['LAT'],
        mode='markers',
        text=current_df["Body of Water Name"],
        visible=True,
        # name = "MC > WHO Limit",
        marker=dict(
            size=6,
            reversescale=True,
            autocolorscale=False,
            symbol='circle',
            opacity=0.6,
            line=dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale='Viridis',
            cmin=0,
            color=selected_data['MC_pc_bin'],
            cmax=selected_data['MC_pc_bin'].max(),
            colorbar=dict(
                title="Value")
        ))]

    layout = go.Layout(title='Log Microcystin Concentration Change',
                       showlegend=False,
                       geo=dict(
                           scope='world',
                           showframe=False,
                           showcoastlines=True,
                           showlakes=True,
                           showland=True,
                           landcolor="rgb(229, 229, 229)",
                           showrivers=True
                       ))

    fig = go.Figure(layout=layout, data=data)
    return fig


def geo_concentration_plot(selected_data):
    data = []
    opacity_level = 0.8
    MC_conc = selected_data['Microcystin (ug/L)']
    # make bins
    b1 = selected_data[MC_conc <= USEPA_LIMIT]
    b2 = selected_data[(MC_conc > USEPA_LIMIT) & (MC_conc <= WHO_LIMIT)]
    b3 = selected_data[MC_conc > WHO_LIMIT]
    data.append(go.Scattergeo(
        lon=b1['LONG'],
        lat=b1['LAT'],
        mode='markers',
        text=b1["Body of Water Name"],
        visible=True,
        name="MC <= USEPA Limit",
        marker=dict(color="green", opacity=opacity_level)))
    data.append(go.Scattergeo(
        lon=b2['LONG'],
        lat=b2['LAT'],
        mode='markers',
        text=b2["Body of Water Name"],
        visible=True,
        name="MC <= WHO Limit",
        marker=dict(color="orange", opacity=opacity_level)))
    data.append(go.Scattergeo(
        lon=b3['LONG'],
        lat=b3['LAT'],
        mode='markers',
        text=b3["Body of Water Name"],
        visible=True,
        name="MC > WHO Limit",
        marker=dict(color="red", opacity=opacity_level)))

    layout = go.Layout(showlegend=True,
                       hovermode='closest',
                       title="Microcystin Concentration",
                       geo=dict(
                           scope='world',
                           showframe=False,
                           showcoastlines=True,
                           showlakes=True,
                           showland=True,
                           landcolor="rgb(229, 229, 229)",
                           showrivers=True
                       ))

    fig = go.Figure(layout=layout, data=data)
    return fig


def geo_plot(selected_years, selected_month, geo_option, current_df):
    if type(selected_years) is not list:
        selected_years = [selected_years]

    month = pd.to_datetime(current_df['DATETIME']).dt.month
    year = pd.to_datetime(current_df['DATETIME']).dt.year
    selected_data = current_df[(month.isin(selected_month)) & (year.isin(selected_years))]
    if geo_option == "CONC":
        return geo_concentration_plot(selected_data)
    else:
        return geo_log_plot(selected_data, current_df)


def tn_tp(tn_val, tp_val, current_df):
    min_tn = tn_val[0]
    max_tn = tn_val[1]
    min_tp = tp_val[0]
    max_tp = tp_val[1]

    if max_tn == 0:
        max_tn = np.max(current_df["Total Nitrogen (ug/L)"])

    if max_tp == 0:
        max_tp = np.max(current_df["Total Phosphorus (ug/L)"])

    dat = current_df[
        (current_df["Total Nitrogen (ug/L)"] >= min_tn) & (current_df["Total Nitrogen (ug/L)"] <= max_tn) & (
                current_df["Total Phosphorus (ug/L)"] >= min_tp) & (
                current_df["Total Phosphorus (ug/L)"] <= max_tp)]
    MC_conc = dat['Microcystin (ug/L)']
    # make bins
    b1 = dat[MC_conc <= USEPA_LIMIT]
    b2 = dat[(MC_conc > USEPA_LIMIT) & (MC_conc <= WHO_LIMIT)]
    b3 = dat[MC_conc > WHO_LIMIT]

    data = [go.Scatter(
        x=np.log(b1["Total Nitrogen (ug/L)"]),
        y=np.log(b1["Total Phosphorus (ug/L)"]),
        mode='markers',
        name="<USEPA",
        text=current_df["Body of Water Name"],
        marker=dict(
            size=8,
            color="green",  # set color equal to a variable
        )),
        go.Scatter(
            x=np.log(b2["Total Nitrogen (ug/L)"]),
            y=np.log(b2["Total Phosphorus (ug/L)"]),
            mode='markers',
            name=">USEPA",
            text=current_df["Body of Water Name"],
            marker=dict(
                size=8,
                color="orange"  # set color equal to a variable
            )),
        go.Scatter(
            x=np.log(b3["Total Nitrogen (ug/L)"]),
            y=np.log(b3["Total Phosphorus (ug/L)"]),
            mode='markers',
            name=">WHO",
            text=current_df["Body of Water Name"],
            marker=dict(
                size=8,
                color="red",  # set color equal to a variable
            ))]

    layout = go.Layout(
        showlegend=True,
        xaxis=dict(
            title='log TN'),
        yaxis=dict(
            title="log TP"),
        hovermode='closest'
    )

    return (go.Figure(data=data, layout=layout))


def correlation_plot(selected_dataset, current_df):
    # IN PROGRESS
    # selected_col_stripped = re.sub("[\(\[].*?[\)\]]", "", selected_col)
    # selected_col_stripped = re.sub('\s+', ' ', selected_col_stripped).strip()

    selected_data = current_df['DATETIME', selected_dataset]

    # calculate correlation coefficient for each point as the z data

    # x_data = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    # y_data = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    # z_data = ["morning", "afternoon", "evening"]

    data = go.Heatmap(
        x=selected_data,
        y=selected_data,
        z=z_data
    )
    layout = go.Layout(
        title='%s vs Date' % selected_x,  # stripped
        # xaxis={'title':'Date'},
        # yaxis={'title': str(selected_x)}
    )
    correlation_plot = {
        'data': data,
        'layout': layout
    }
    return correlation_plot


def comparison_plot(selected_y, selected_x, current_df):
    selected_data = current_df[[selected_y, selected_x]]

    x_data = selected_data[selected_x]
    y_data = selected_data[selected_y]

    data = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers')

    layout = go.Layout(
        title='%s vs %s' % (selected_y, selected_x),
        xaxis={'title': str(selected_x)},
        yaxis={'title': str(selected_y)},
        hovermode='closest'
    )

    comparison_plot = {
        'data': [data],
        'layout': layout
    }
    return comparison_plot


def temporal_lake(selected_col, selected_loc, selected_type, current_df):
    selected_col_stripped = re.sub("[\(\[].*?[\)\]]", "", selected_col)
    selected_col_stripped = re.sub('\s+', ' ', selected_col_stripped).strip()

    selected_data = current_df[current_df['Body of Water Name'] == selected_loc]
    x_data = pd.to_datetime(selected_data['DATETIME'])
    print(len(selected_data[selected_col]))

    if len(selected_data[selected_col]) >= 3:
        if selected_type == 'raw':
            y_data = selected_data[selected_col]
            print(len(y_data))
            title = '%s Trends' % (selected_col_stripped)
            y_axis = str(selected_col)
        else:
            y_data = selected_data[selected_col].pct_change()
            title = 'Percent Change in %s Trends' % (selected_col_stripped)
            y_axis = 'Percent Change in %s' % (selected_col_stripped)
    else:
        title = ''
        y_data = []
        y_axis = ''

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': y_axis},
        hovermode='closest'
    )
    temporal_lake_plot = plot_line(x_data, y_data, layout)

    return temporal_lake_plot


def temporal_overall(selected_col, selected_type, current_df):
    selected_col_stripped = re.sub("[\(\[].*?[\)\]]", "", selected_col)
    selected_col_stripped = re.sub('\s+', ' ', selected_col_stripped).strip()
    selected_data = current_df[['DATETIME', selected_col]]
    months = pd.to_datetime(selected_data['DATETIME']).dt.to_period("M")
    selected_data_month = selected_data.groupby(months)
    selected_data_month = selected_data_month.agg(['mean'])
    x_data = selected_data_month.index.to_timestamp()

    if selected_type == 'avg':
        y_data = selected_data_month[selected_col]['mean']
        title = '%s vs Date' % selected_col_stripped
        y_axis = str(selected_col)
    else:
        y_data = selected_data_month[selected_col]['mean'].pct_change()
        title = 'Percent Change of %s vs Date' % selected_col_stripped
        y_axis = 'Percent Change of %s' % selected_col_stripped

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': y_axis},
        hovermode='closest'
    )
    temporal_overall_plot = plot_line(x_data, y_data, layout)

    return temporal_overall_plot


def temporal_raw(selected_option, selected_col, log_range, current_df):
    min_log = log_range[0]
    max_log = log_range[1]

    if max_log == 0:
        max_log = np.max(current_df[selected_col])

    dat = current_df[(current_df[selected_col] >= min_log) & (current_df[selected_col] <= max_log)]
    MC_conc = dat['Microcystin (ug/L)']

    selected_col_stripped = re.sub("[\(\[].*?[\)\]]", "", selected_col)
    selected_col_stripped = re.sub('\s+', ' ', selected_col_stripped).strip()
    selected_data = current_df[['DATETIME', selected_col]]

    if selected_option == '3SD':
        selected_data = selected_data[((selected_data[selected_col] - selected_data[selected_col].mean()) /
                                       selected_data[selected_col].std()).abs() < 3]
    x_data = selected_data['DATETIME']
    y_data = MC_conc
    if selected_option == 'LOG':
        y_data = np.log(MC_conc)

    layout = go.Layout(
        title='%s vs Date' % selected_col_stripped,
        xaxis={'title': 'Date'},
        yaxis={'title': str(selected_col)},
        hovermode='closest'
    )

    data = go.Scatter(
        x=x_data,
        y=y_data,
        text="Lake: " + current_df["Body of Water Name"],
        mode='markers',
        marker={
            'opacity': 0.8,
        },
        line={
            'width': 1.5
        }
    )

    temporal_raw_plot = {
        'data': [data],
        'layout': layout
    }
    return temporal_raw_plot


def plot_line(x_data, y_data, layout):
    data = go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        marker={
            'opacity': 0.8,
        },
        line={
            'width': 1.5
        }
    )
    fig = {
        'data': [data],
        'layout': layout
    }
    return fig


# Website layout HTML code
app.layout = html.Div(children=[
    html.Div([
        html.H1(children='GLEON MC Data Analysis')
    ], className="title"),

    html.Div([
        html.Details([
            html.Summary('Upload New Data'),
            html.Div(children=[
                html.H4('How to Upload Data'),
                html.P('1. Download the outline file below and copy the appropriate data into the csv file.'),
                html.P(
                    '2. Fill out the metadata questionnaire below with appropriate information and links as needed.'),
                html.P('3. Select or drag and drop the filled out csv file containing your data.'),
                html.P('4. Click \'Upload\' to upload your data and information to the project.'),
                # TODO: datasheet outline is required for db to parse data correctly from spreadsheet, need to make this clearer to
                # users when we go live
                html.A('Download Datasheet Outline File',
                       id='example-outline-link',
                       href='assets/GLEON_GMA_Example.xlsx',
                       target='_blank',
                       download='GLEON_GMA_Example.xlsx')
            ], className="row"),

            # Upload New Data questionnaire
            html.Div(children=[
                html.Div([
                    html.Div([
                        html.P('Name'),
                        dcc.Input(id='user-name', type='text'),
                    ], className='one-third column'),
                    html.Div([
                        html.P('Institution'),
                        dcc.Input(id='user-inst', type='text'),
                    ], className='one-third column'),
                    html.Div([
                        html.P('Database Name'),
                        dcc.Input(id='db-name', type='text')
                    ], className='one-third column'),
                ], className="row"),

                html.P('Is the data peer reviewed or published?'),
                dcc.RadioItems(
                    id="is-data-reviewed",
                    options=[{'label': 'Yes', 'value': 'is-reviewed'},
                             {'label': 'No', 'value': 'not-reviewed'}],
                ),
                # if answered yes, URL link field appears for user to submit an appropriate link
                dcc.Input(
                    placeholder='URL Link',
                    type='text',
                    value='',
                    id='publication-url',
                    style={'display': 'none'}
                ),
                html.P('Is the field method reported?'),
                dcc.RadioItems(
                    id="is-field-method-reported",
                    options=[{'label': 'Yes', 'value': 'fm-reported'},
                             {'label': 'No', 'value': 'fm-not-reported'}],
                ),
                dcc.Input(
                    placeholder='URL Link',
                    type='text',
                    value='',
                    id='field-method-report-url',
                    style={'display': 'none'}
                ),
                html.P('Is the lab method reported?'),
                dcc.RadioItems(
                    id="is-lab-method bui-reported",
                    options=[{'label': 'Yes', 'value': 'lm-reported'},
                             {'label': 'No', 'value': 'lm-not-reported'}],
                ),
                dcc.Input(
                    placeholder='URL Link',
                    type='text',
                    value='',
                    id='lab-method-report-url',
                    style={'display': 'none'}
                ),
                html.P('Is the QA/QC data available?'),
                dcc.RadioItems(
                    id="is-qaqc-available",
                    options=[{'label': 'Yes', 'value': 'qaqc-available'},
                             {'label': 'No', 'value': 'qaqc-not-available'}],
                ),
                dcc.Input(
                    placeholder='URL Link',
                    type='text',
                    value='',
                    id='qaqc-url',
                    style={'display': 'none'}
                ),
                html.P('Is the full QA/QC data available upon request?'),
                dcc.RadioItems(
                    id="is-full-qaqc-available",
                    options=[{'label': 'Yes', 'value': 'full-qaqc-available'},
                             {'label': 'No', 'value': 'full-qaqc-not-available'}],
                ),
                dcc.Input(
                    placeholder='URL Link',
                    type='text',
                    value='',
                    id='full-qaqc-url',
                    style={'display': 'none'}
                ),

                html.P('Substrate'),
                dcc.Dropdown(
                    id='substrate-option',
                    multi=False,
                    options=[{'label': 'Planktonic', 'value': 'planktonic'},
                             {'label': 'Beach', 'value': 'beach'},
                             {'label': 'Periphyton', 'value': 'periphyton'}],
                    style={
                        'margin': '0 60px 0 0',
                        'width': '95%'
                    }
                ),
                html.P('Sample Types'),
                dcc.Dropdown(
                    id='sample-type-option',
                    multi=False,
                    options=[{'label': 'Routine Monitoring', 'value': 'routine-monitoring'},
                             {'label': 'Reactionary Water Column', 'value': 'reactionary-water-column'},
                             {'label': 'Scum Focused', 'value': 'scum-focused'}],
                    style={
                        'margin': '0 60px 0 0',
                        'width': '95%'
                    }
                ),
                html.P('Field Methods'),
                dcc.Dropdown(
                    id='field-method-option',
                    multi=False,
                    options=[{'label': 'Vertically Integrated Sample', 'value': 'vertically-integrated'},
                             {'label': 'Discrete Depth Sample', 'value': 'discrete-depth'},
                             {'label': 'Spatially Integrated Sample', 'value': 'spatially-integrated'}],
                    style={
                        'margin': '0 60px 10px 0',
                        'width': '95%'
                    }
                ),
                dcc.Input(
                    placeholder='Depth Integrated (m)',
                    type='text',
                    id='vertically-depth-integrated',
                    style={'display': 'none'}
                ),
                dcc.Input(
                    placeholder='Depth Sampled (m)',
                    type='text',
                    id='discrete-depth-sampled',
                    style={'display': 'none'}
                ),
                dcc.Input(
                    placeholder='Depth of Sample (m)',
                    type='text',
                    id='spatially-integrated-depth',
                    style={'display': 'none'}
                ),
                dcc.Input(
                    placeholder='# of samples integrated',
                    type='text',
                    id='num-spatially-integrated-samples',
                    style={'display': 'none'}
                ),
                html.P('Microcystin Method'),
                dcc.Dropdown(
                    id='microcystin-method',
                    multi=False,
                    options=[
                        {'label': 'PPIA', 'value': 'PPIA'},
                        {'label': 'ELISA', 'value': 'ELISA'},
                        {'label': 'LC-MSMS', 'value': 'LC-MSMS'}
                    ],
                    style={
                        'margin': '0 60px 0 0',
                        'width': '95%'
                    }
                ),
                html.P('Was Sample Filtered?'),
                dcc.RadioItems(
                    id="sample-filtered",
                    options=[{'label': 'Yes', 'value': 'is-filtered'},
                             {'label': 'No', 'value': 'not-filtered'}]
                ),
                dcc.Input(
                    placeholder='Filter Size (μm)',
                    type='text',
                    id='filter-size',
                    style={'display': 'none'}
                ),
                html.P('Cell count method?'),
                dcc.Input(
                    placeholder='URL Link',
                    type='text',
                    value='',
                    id='cell-count-url',
                ),
                html.P('Ancillary data available?'),
                dcc.Textarea(
                    id='ancillary-data',
                    placeholder='Description of parameters or URL link'
                ),

                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Database File')
                    ]),
                    style={
                        'width': '90%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '25px 0 0 0',
                    },
                    # allow single file upload
                    multiple=False
                ),
                html.Div(id='upload-output'),
                html.Button(id='upload-button', n_clicks=0, children='Upload',
                            style={
                                'margin': '15px 0px 10px 0px'
                            }
                            ),
                html.P(id='upload-msg'),
            ], className="row p"),
        ]),
    ], className="row"),

    html.Button(id='refresh-db-button', children='Refresh',
                style={
                    'margin': '10px 0px 10px 0px'
                }
                ),

    dash_table.DataTable(
        id='metadata_table',
        columns=[
            # the column names are seen in the UI but the id should be the same as dataframe col name
            # the DB ID column is hidden - later used to find DB pkl files in the filtering process
            # TODO: add column for field method in table
            {'name': 'Database ID', 'id': 'DB_ID'},
            {'name': 'Database Name', 'id': 'DB_name'},
            {'name': 'Uploaded By', 'id': 'Uploaded_by'},
            {'name': 'Upload Date', 'id': 'Upload_date'},
            {'name': 'Microcystin Method', 'id': 'Microcystin_method'},
            {'name': 'Number of Lakes', 'id': 'N_lakes'},
            {'name': 'Number of Samples', 'id': 'N_samples'}, ],
        data=get_metadata_table_content(metadataDB),
        row_selectable='multi',
        selected_rows=[],
        style_as_list_view=True,
        # sorting=True,
        style_table={'overflowX': 'scroll'},
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
    ),

    html.Button(id='apply-filters-button', children='Filter Data',
                style={
                    'margin': '10px 0px 10px 0px'
                }
                ),
    # Export the selected datasets in a single csv file
    html.A(html.Button(id='export-data-button', children='Download Filtered Data',
                       style={
                           'margin': '10px 0px 10px 10px'
                       }),
           href='',
           id='download-link',
           download='data.csv',
           target='_blank'
           ),
    # Geographical world map showing concentration locations
    html.Div([
        html.H2('Microcystin Concentration'),
        dcc.Graph(id='geo_plot'),
        html.Div([
            dcc.RadioItems(
                id="geo_plot_option",
                options=[{'label': 'Show Concentration Plot', 'value': 'CONC'},
                         {'label': 'Show Log Concentration Change Plot', 'value': 'LOG'}],
                value='CONC'),
        ]),
        html.Div([
            html.Div(html.P("Year:")),
            html.Div(
                dcc.Dropdown(
                    id='year-dropdown',
                    multi=True
                ),
            )
        ]),
        html.Div([
            html.P("Month:"),
            dcc.RangeSlider(
                id='month-slider',
                min=0,
                max=11,
                value=[0, 11],
                # included=False,
                marks={i: months[i] for i in range(len(months))}
            )
        ]),
    ], className="row"),

    html.Div([
        html.H2('Raw Data'),
        dcc.Graph(
            id="temporal-raw-scatter",
        ),
        html.Div([
            html.Div([
                html.P("Y-Axis Range"),
                dcc.RangeSlider(
                    id="axis_range_raw",
                    min=0,
                    step=0.5,
                    marks={
                        1: '1',
                        100: '100',
                        1000: '1000',
                        10000: '10000'},
                ),
            ], style={'marginBottom': 30}),
        ]),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="temporal-raw-col"
                )], className='six columns'),
            html.Div([
                dcc.RadioItems(
                    id="temporal-raw-option",
                    options=[{'label': 'Show All Raw Data', 'value': 'RAW'},
                             {'label': 'Apply Log10 to Raw Data', 'value': 'LOG'},
                             {'label': 'Show Data Within 3 Standard Deviations', 'value': '3SD'}],
                    value='RAW')
            ], className='six columns'),
        ])
    ], className='row'),

    # comparison graph between two selected categories for the selected data
    html.Div([
        dcc.Graph(
            id="comparison_scatter",
        ),
        html.Div([
            html.Div([
                html.P('Y-axis'),
                dcc.Dropdown(
                    id='compare-y-axis',
                )], className='six columns'),
            html.Div([
                html.P('X-axis'),
                dcc.Dropdown(
                    id='compare-x-axis')
            ], className='six columns')
        ])
    ], className='row'),

    # INCOMPLETE: correlation matrix for a single dataset to show a heatmap of correlations (Michael will finish this)
    # html.Div([
    #     html.H2('Correlation Matrix'),
    #     dcc.Graph(id='correlation-graph'),
    #     html.Div([
    #         html.Div([
    #             html.P('Dataset'),
    #             dcc.Dropdown(
    #                 id='correlation-dropdown'),
    #             ], className='six columns')
    #         ])
    #     ], className='row'),

    html.Div([
        html.H2('Total Phosphorus vs Total Nitrogen'),
        dcc.Graph(
            id="tn_tp_scatter",
        ),
        html.Div([
            html.P("Log TN:"),
            dcc.RangeSlider(
                id="tn_range",
                min=0,
                step=0.5,
                marks={
                    1000: '1',
                    4000: '100',
                    7000: '1000',
                    10000: '10000'
                },
            ),
        ]),
        html.Div([
            html.P("Log TP:"),
            dcc.RangeSlider(
                id="tp_range",
                min=0,
                step=0.5,
                marks={
                    1000: '1',
                    4000: '100',
                    7000: '1000',
                    10000: '10000'
                },
            ),
        ]),
    ], className="row"),

    html.Div([
        html.H2('Data Trends by Lake'),
        html.P('Lakes require at least three data points to have a trendline', id='lake-minimum-points'),
        html.Div([
            html.Div([
                dcc.Graph(
                    id="temporal-lake-scatter",
                )
            ], className='six columns'),
            html.Div([
                dcc.Graph(
                    id="temporal-lake-pc-scatter",
                )
            ], className='six columns'),
        ]),
        dcc.Dropdown(
            id="temporal-lake-col",
            className='six columns'
        ),
        dcc.Dropdown(
            id='temporal-lake-location',
            className='six columns'
        )
    ], className="row"),

    html.Div([
        html.H2('Overall Temporal Data Trends'),
        html.P('Includes data from all lakes'),
        html.Div([
            html.Div([
                dcc.Graph(
                    id="temporal-avg-scatter",
                )
            ], className='six columns'),
            html.Div([
                dcc.Graph(
                    id="temporal-pc-scatter",
                )
            ], className='six columns'),
        ]),
        dcc.Dropdown(
            id="temporal-avg-col"
        )
    ], className='row'),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'}, children=convert_to_json(empty_df))
])


# Controls if text fields are visible based on selected options in upload questionnaire
@app.callback(
    dash.dependencies.Output('publication-url', 'style'),
    [dash.dependencies.Input('is-data-reviewed', 'value')]
)
def show_peer_review_url(is_peer_reviewed):
    if is_peer_reviewed == 'is-reviewed':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('field-method-report-url', 'style'),
    [dash.dependencies.Input('is-field-method-reported', 'value')]
)
def show_field_method_url(is_fm_reported):
    if is_fm_reported == 'fm-reported':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('lab-method-report-url', 'style'),
    [dash.dependencies.Input('is-lab-method bui-reported', 'value')]
)
def show_lab_method_url(is_lm_reported):
    if is_lm_reported == 'lm-reported':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('qaqc-url', 'style'),
    [dash.dependencies.Input('is-qaqc-available', 'value')]
)
def show_qaqc_url(is_qaqc_available):
    if is_qaqc_available == 'qaqc-available':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('full-qaqc-url', 'style'),
    [dash.dependencies.Input('is-full-qaqc-available', 'value')]
)
def show_full_qaqc_url(is_full_qaqc_available):
    if is_full_qaqc_available == 'full-qaqc-available':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    [dash.dependencies.Output('vertically-depth-integrated', 'style'),
     dash.dependencies.Output('discrete-depth-sampled', 'style'),
     dash.dependencies.Output('spatially-integrated-depth', 'style'),
     dash.dependencies.Output('num-spatially-integrated-samples', 'style')],
    [dash.dependencies.Input('field-method-option', 'value')])
def show_field_option_input(field_option):
    if field_option == 'vertically-integrated':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    if field_option == 'discrete-depth':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    if field_option == 'spatially-integrated':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    dash.dependencies.Output('filter-size', 'style'),
    [dash.dependencies.Input('sample-filtered', 'value')]
)
def show_filter_size(visibility_state):
    if visibility_state == 'is-filtered':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('metadata_table', 'data'),
    [dash.dependencies.Input('refresh-db-button', 'n_clicks')])
def upload_file(n_clicks):
    # read from MetadataDB to update the table
    metadataDB = pd.read_csv("MetadataDB.csv")
    return get_metadata_table_content(metadataDB)


@app.callback(
    dash.dependencies.Output('geo_plot', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value'),
     dash.dependencies.Input('month-slider', 'value'),
     dash.dependencies.Input('geo_plot_option', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_geo_plot(selected_years, selected_month, geo_option, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return geo_plot(selected_years, selected_month, geo_option, dff)


@app.callback(
    dash.dependencies.Output('comparison_scatter', 'figure'),
    [dash.dependencies.Input('compare-y-axis', 'value'),
     dash.dependencies.Input('compare-x-axis', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_comparison(selected_y, selected_x, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return comparison_plot(selected_y, selected_x, dff)


# TODO: Correlation matrix in progress
# @app.callback(
#     dash.dependencies.Output('correlation-graph', 'figure'),
#     [dash.dependencies.Input('correlation-dropdown', 'value'),
#     dash.dependencies.Input('intermediate-value', 'children')])
# def update_correlation(selected_dataset, jsonified_data):
#     dff = convert_to_df(jsonified_data)
#     return correlation_plot(selected_dataset, dff)

@app.callback(
    dash.dependencies.Output('temporal-lake-scatter', 'figure'),
    [dash.dependencies.Input('temporal-lake-col', 'value'),
     dash.dependencies.Input('temporal-lake-location', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_temporal_output(selected_col, selected_loc, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return temporal_lake(selected_col, selected_loc, 'raw', dff)


@app.callback(
    dash.dependencies.Output('temporal-lake-pc-scatter', 'figure'),
    [dash.dependencies.Input('temporal-lake-col', 'value'),
     dash.dependencies.Input('temporal-lake-location', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_output(selected_col, selected_loc, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return temporal_lake(selected_col, selected_loc, 'pc', dff)


@app.callback(
    dash.dependencies.Output('tn_tp_scatter', 'figure'),
    [dash.dependencies.Input('tn_range', 'value'),
     dash.dependencies.Input('tp_range', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_output(tn_val, tp_val, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return tn_tp(tn_val, tp_val, dff)


@app.callback(
    dash.dependencies.Output('temporal-avg-scatter', 'figure'),
    [dash.dependencies.Input('temporal-avg-col', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_output(selected_col, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return temporal_overall(selected_col, 'avg', dff)


@app.callback(
    dash.dependencies.Output('temporal-pc-scatter', 'figure'),
    [dash.dependencies.Input('temporal-avg-col', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')])
def update_output(selected_col, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return temporal_overall(selected_col, 'pc', dff)


@app.callback(
    dash.dependencies.Output('temporal-raw-scatter', 'figure'),
    [dash.dependencies.Input('temporal-raw-option', 'value'),
     dash.dependencies.Input('temporal-raw-col', 'value'),
     dash.dependencies.Input('axis_range_raw', 'value'),
     dash.dependencies.Input('intermediate-value', 'children')
     ])
def update_output(selected_option, selected_col, log_range, jsonified_data):
    dff = convert_to_df(jsonified_data)
    return temporal_raw(selected_option, selected_col, log_range, dff)


@app.callback(dash.dependencies.Output('upload-output', 'children'),
              [dash.dependencies.Input('upload-data', 'contents')],
              [dash.dependencies.State('upload-data', 'filename')])
def update_uploaded_file(contents, filename):
    if contents is not None:
        return html.Div([
            html.H6(filename),
        ])


@app.callback(
    dash.dependencies.Output('upload-msg', 'children'),
    [dash.dependencies.Input('upload-button', 'n_clicks')],
    [dash.dependencies.State('db-name', 'value'),
     dash.dependencies.State('user-name', 'value'),
     dash.dependencies.State('user-inst', 'value'),
     dash.dependencies.State('upload-data', 'contents'),
     dash.dependencies.State('upload-data', 'filename'),
     dash.dependencies.State('publication-url', 'value'),
     dash.dependencies.State('field-method-report-url', 'value'),
     dash.dependencies.State('lab-method-report-url', 'value'),
     dash.dependencies.State('qaqc-url', 'value'),
     dash.dependencies.State('full-qaqc-url', 'value'),
     dash.dependencies.State('substrate-option', 'value'),
     dash.dependencies.State('sample-type-option', 'value'),
     dash.dependencies.State('field-method-option', 'value'),
     dash.dependencies.State('microcystin-method', 'value'),
     dash.dependencies.State('filter-size', 'value'),
     dash.dependencies.State('cell-count-url', 'value'),
     dash.dependencies.State('ancillary-data', 'value')])
def upload_file(n_clicks, dbname, username, userinst, contents, filename, publicationURL, fieldMURL, labMURL, QAQCUrl,
                fullQAQCUrl, substrate, sampleType, fieldMethod, microcystinMethod, filterSize, cellCountURL,
                ancillaryURL):
    if n_clicks != None and n_clicks > 0:
        if username == None or not username.strip():
            return 'Name field cannot be empty.'
        elif userinst == None or not userinst.strip():
            return 'Institution cannot be empty.'
        elif dbname == None or not dbname.strip():
            return 'Database name cannot be empty.'
        elif contents is None:
            return 'Please select a file.'
        else:
            new_db = db_info(dbname, username, userinst)
            new_db.db_publication_url = publicationURL
            new_db.db_field_method_url = fieldMURL
            new_db.db_lab_method_url = labMURL
            new_db.db_QAQC_url = QAQCUrl
            new_db.db_full_QAQC_url = fullQAQCUrl
            new_db.db_substrate = substrate
            new_db.db_sample_type = sampleType
            new_db.db_field_method = fieldMethod
            new_db.db_microcystin_method = microcystinMethod
            new_db.db_filter_size = filterSize
            new_db.db_cell_count_method = cellCountURL
            new_db.db_ancillary_url = ancillaryURL

            return upload_new_database(new_db, contents, filename)


@app.callback(
    [dash.dependencies.Output('intermediate-value', 'children'),
     dash.dependencies.Output('tn_range', 'max'),
     dash.dependencies.Output('tn_range', 'value'),
     dash.dependencies.Output('tp_range', 'max'),
     dash.dependencies.Output('tp_range', 'value'),
     dash.dependencies.Output('year-dropdown', 'options'),
     dash.dependencies.Output('year-dropdown', 'value'),
     dash.dependencies.Output('temporal-lake-location', 'options'),
     dash.dependencies.Output('temporal-lake-location', 'value'),
     dash.dependencies.Output('temporal-lake-col', 'options'),
     dash.dependencies.Output('temporal-lake-col', 'value'),
     dash.dependencies.Output('temporal-avg-col', 'options'),
     dash.dependencies.Output('temporal-avg-col', 'value'),
     dash.dependencies.Output('temporal-raw-col', 'options'),
     dash.dependencies.Output('temporal-raw-col', 'value'),
     dash.dependencies.Output('axis_range_raw', 'max'),
     dash.dependencies.Output('axis_range_raw', 'value'),
     dash.dependencies.Output('compare-y-axis', 'options'),
     dash.dependencies.Output('compare-y-axis', 'value'),
     dash.dependencies.Output('compare-x-axis', 'options'),
     dash.dependencies.Output('compare-x-axis', 'value')],
    # dash.dependencies.Output('correlation-dropdown', 'options'),
    # dash.dependencies.Output('correlation-dropdown', 'value')] -- for correlation matrix
    [dash.dependencies.Input('apply-filters-button', 'n_clicks')],
    [dash.dependencies.State('metadata_table', 'derived_virtual_selected_rows'),
     dash.dependencies.State('metadata_table', 'derived_virtual_data')])
def update_graph(n_clicks, derived_virtual_selected_rows, dt_rows):
    if n_clicks != None and n_clicks > 0 and derived_virtual_selected_rows is not None:
        # update the user's data based on the selected databases
        selected_rows = [dt_rows[i] for i in derived_virtual_selected_rows]
        new_df = update_dataframe(selected_rows)
        print("NEW DF: ", new_df)

        # List of datasets and notice for correlation matrix
        correlation_notice = {'display': 'block'}
        db_name = [{'label': row['DB_name'], 'value': row['DB_name']} for row in selected_rows]
        db_value = db_name[0]

        jsonStr1 = convert_to_json(new_df)

        # update range for raw data graph
        raw_range_max = np.max(new_df["Microcystin (ug/L)"])
        raw_range_value = [0, np.max(new_df["Microcystin (ug/L)"])]

        tn_max = np.max(new_df["Total Nitrogen (ug/L)"])
        tn_value = [0, np.max(new_df["Total Nitrogen (ug/L)"])]

        tp_max = np.max(new_df["Total Phosphorus (ug/L)"])
        tp_value = [0, np.max(new_df["Total Phosphorus (ug/L)"])]

        # update the date ranges
        year = pd.to_datetime(new_df['DATETIME']).dt.year
        years = range(np.min(year), np.max(year) + 1)
        years_options = [{'label': str(y), 'value': y} for y in years]

        # update the lake locations
        locs = list(new_df["Body of Water Name"].unique())
        locs.sort()
        locs_options = [{'label': loc, 'value': loc} for loc in locs]
        locs_value = locs[0]

        # get current existing column names and remove general info to update the dropdowns of plot axes
        colNames = new_df.columns.values.tolist()
        if 'DATETIME' in colNames: colNames.remove('DATETIME')
        if 'Body of Water Name' in colNames: colNames.remove('Body of Water Name')
        if 'DataContact' in colNames: colNames.remove('DataContact')
        if 'LONG' in colNames: colNames.remove('LONG')
        if 'LAT' in colNames: colNames.remove('LAT')
        if 'Comments' in colNames: colNames.remove('Comments')
        if 'MC Percent Change' in colNames: colNames.remove('MC Percent Change')
        if 'Maximum Depth (m)' in colNames: colNames.remove('Maximum Depth (m)')
        if 'Mean Depth (m)' in colNames: colNames.remove('Mean Depth (m)')

        colNames.sort()
        col_options = [{'label': col, 'value': col} for col in colNames]
        col_value = colNames[0]
        col_value_next = colNames[1]

        return jsonStr1, tn_max, tn_value, tp_max, tp_value, years_options, years_options, locs_options, locs_value, col_options, col_value, col_options, col_value, col_options, col_value, raw_range_max, raw_range_value, col_options, col_value, col_options, col_value_next,  # db_name, db_value


# Update the download link to contain the data from the selected datasheets
@app.callback(
    dash.dependencies.Output('download-link', 'href'),
    [dash.dependencies.Input('apply-filters-button', 'n_clicks')],
    [dash.dependencies.State('metadata_table', 'derived_virtual_selected_rows'),
     dash.dependencies.State('metadata_table', 'derived_virtual_data')])
def update_data_download_link(n_clicks, derived_virtual_selected_rows, dt_rows):
    if n_clicks != None and n_clicks > 0 and derived_virtual_selected_rows is not None:
        selected_rows = [dt_rows[i] for i in derived_virtual_selected_rows]
        dff = update_dataframe(selected_rows)

        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "//fonts.googleapis.com/css?family=Dosis:Medium",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/62f0eb4f1fadbefea64b2404493079bf848974e8/dash-uber-ride-demo.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://codepen.io/chriddyp/pen/bWLwgP.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=True)
