from django.shortcuts import render
from django.http import HttpResponse
from .forms import CSVUploadForm
import pandas as pd
import joblib  # For model saving/loading
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import numpy as np

def home(request):
    form = CSVUploadForm()

    if request.method == 'POST':
        action = request.POST.get('action')  # Get which button was pressed

        if request.FILES:
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                csv_file = form.cleaned_data['csv_file']
                # Read the CSV file into a pandas DataFrame
                csv_data = pd.read_csv(csv_file)
                csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'])
                request.session['csv_data'] = csv_data.to_json(date_format='iso')
                return render(request, 'home.html', {
                    'form': form, 
                    'success': 'File uploaded successfully!',
                    'p': request.POST.get('p', 0),
                    'q': request.POST.get('q', 0),
                    'd': request.POST.get('d', 0),
                    })

        elif action == 'train':
            # Retrieve CSV data from session
            csv_data_json = request.session.get('csv_data')
            if csv_data_json:
                csv_data = pd.read_json(csv_data_json, convert_dates=['timestamp'])

                # Sort the DataFrame by the timestamp to ensure chronological order
                csv_data = csv_data.sort_values(by='timestamp').reset_index(drop=True)

                # Get ARIMA parameters from the form
                try:
                        p = int(request.POST.get('p'))
                        d = int(request.POST.get('d'))
                        q = int(request.POST.get('q'))
                except (ValueError, TypeError):
                    return render(request, 'home.html', {
                        'form': form,
                        'error': 'Please enter valid integer values for P, D, and Q.',
                        'p': request.POST.get('p', 0),
                        'd': request.POST.get('d', 0),
                        'q': request.POST.get('q', 0),
                    })
                
                # Train/test split 
                train_size = int(len(csv_data) * 0.8)
                train, test = csv_data['value'][:train_size], csv_data['value'][train_size:]

                # Train ARIMA model
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()

                request.session['train_size'] = train_size
                # Save model using joblib
                joblib.dump(model_fit, 'trained_arima_model.pkl')

                return render(request, 'home.html', {
                    'form': form,
                    'success': 'Model training complete!',
                    'p': p,
                    'd': d,
                    'q': q,
                })

            return render(request, 'home.html', {
                'form': form,
                'error': 'No CSV data found. Please upload the CSV first.',
                'p': request.POST.get('p', 0),
                'd': request.POST.get('d', 0),
                'q': request.POST.get('q', 0),
            })

        elif action == 'analyze':
            # Add logic to analyze the trained model here if needed
            return render(request, 'home.html', {
                'form': form,
                'success': 'Analyze function called!',
                'p': request.POST.get('p', 0),
                'd': request.POST.get('d', 0),
                'q': request.POST.get('q', 0),
            })
        
        elif action == 'predict':
            # Handle the predict action
            return render(request, 'home.html', {
                'form': form,
                'success': 'Predict function called!',
                'p': request.POST.get('p', 0),
                'd': request.POST.get('d', 0),
                'q': request.POST.get('q', 0),
            })

    else:
        form = CSVUploadForm()

    return render(request, 'home.html', {
        'form': form,
        'p': request.POST.get('p', 0),
        'd': request.POST.get('d', 0),
        'q': request.POST.get('q', 0),
    })

def analyze(request):

    csv_data_json = request.session.get('csv_data')
    train_size = request.session.get('train_size')
    if csv_data_json:
        # Convert the JSON back to DataFrame
        csv_data = pd.read_json(csv_data_json, convert_dates=['timestamp'])
        
        # Sort the DataFrame by the timestamp to ensure chronological order
        csv_data = csv_data.sort_values(by='timestamp').reset_index(drop=True)

        # Extract the timestamps and other necessary data
        timestamps = csv_data['timestamp']
        values = csv_data['value']  # Assuming 'value' is your target column
        total_size = int(len(values))
        test_size = total_size - train_size
        # Forecasting steps or any additional logic you need here
        
        # Load the trained model
        model = joblib.load('trained_arima_model.pkl')
        
        # Make predictions or perform analysis
        forecast = model.forecast(steps=test_size)  # For example, forecasting 10 steps ahead

        forecast_index = pd.date_range(start=csv_data['timestamp'].iloc[train_size], periods=test_size, freq='D')
        forecast_df = pd.DataFrame({
            'timestamp': forecast_index,
            'value': forecast,
            'Train/Test': 'Forecast'})
        
        train_data = csv_data.iloc[:train_size+2].copy()
        train_data['Train/Test'] = 'Train'

        test_data = csv_data.iloc[train_size:].copy()
        test_data['Train/Test'] = 'Test'

        combined_data = pd.DataFrame({
            'timestamp': pd.concat([train_data['timestamp'], test_data['timestamp'], forecast_df['timestamp']]),
            'value': pd.concat([train_data['value'], test_data['value'], forecast_df['value']]),
            'Type': ['Train'] * len(train_data) + ['Test'] * len(test_data) + ['Forecast'] * len(forecast_df)
        })
        # Sort by timestamp to ensure the plot is in time order
        combined_data = combined_data.sort_values(by='timestamp').reset_index(drop=True)

        test_values = test_data['value'].values
        rmse = mean_squared_error(test_values, forecast, squared=False)  # RMSE
        mae = mean_absolute_error(test_values, forecast)  # MAE

        # Calculate residuals (test values - forecast values)
        residuals = test_values - forecast

        # Ljung-Box Test for autocorrelation of residuals
        lags = 10  # Set to 10 lags for the test
        ljung_box_test = acorr_ljungbox(residuals, lags=lags, return_df=True)

        # Prepare results for each lag
        ljung_box_results = ljung_box_test[['lb_stat', 'lb_pvalue']]
        ljung_box_results['lag'] = ljung_box_results.index + 1  # Adding lag number

        action = request.POST.get('action')
        if action == 'show_graph':
            # Create the figure using Plotly Express
            fig = px.line(combined_data, x='timestamp', y='value', color='Type',
                        title='Model Analysis: Training, Test Data, and Forecast',
                        labels={'value': 'Value', 'timestamp': 'Timestamp', 'Type': 'Data Type'})

            # Update layout for better readability
            fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=True)),  # Enable range slider
                yaxis_title='Value',
                legend_title='Data Type',
            )

            # Convert th
            plot_html = fig.to_html(full_html=False)
            return render(request, 'analyze.html', {'plot_html': plot_html})
        
        elif action == 'show_error':
            # Return the RMSE, MAE, and Ljung-Box p-value
            return render(request, 'analyze.html', {
                'rmse': rmse, 'mae': mae
            })

        elif action == 'show_residuals_autocorr':
            # ACF of residuals
            autocorr_values = sm.tsa.acf(residuals, nlags=40)  # Autocorrelation up to 40 lags
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(x=list(range(len(autocorr_values))), y=autocorr_values, mode='lines+markers', name='ACF'))
            fig_acf.update_layout(
                title="Autocorrelation of Residuals",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                showlegend=True
            )

            # PACF of residuals
            pacf_values = sm.tsa.pacf(residuals, nlags=40)  # Partial Autocorrelation up to 40 lags
            
            fig_pacf = go.Figure()
            fig_pacf.add_trace(go.Scatter(x=list(range(len(pacf_values))), y=pacf_values, mode='lines+markers', name='PACF'))
            fig_pacf.update_layout(
                title="Partial Autocorrelation of Residuals",
                xaxis_title="Lags",
                yaxis_title="Partial Autocorrelation",
                showlegend=True
            )

            # Combine the two plots into a single HTML to render
            plot_html_acf = fig_acf.to_html(full_html=False)
            plot_html_pacf = fig_pacf.to_html(full_html=False)

            return render(request, 'analyze.html', {
                'plot_html_acf': plot_html_acf,
                'plot_html_pacf': plot_html_pacf
            })
        
        elif action == 'show_residuals_histogram':
            # Calculate the mean of the residuals
            mean_residual = np.mean(residuals)

            # Create a histogram of residuals
            fig = px.histogram(residuals, nbins=30, title="Residuals Histogram",
                            color_discrete_sequence=["#636EFA"])  # Set base color for the bars

            # Update the histogram to add edges
            fig.update_traces(marker=dict(line=dict(color='black', width=1)))  # Set edges to black

            # Update layout
            fig.update_layout(
                xaxis_title="Residuals",
                yaxis_title="Frequency",
                title_font=dict(size=20),
                xaxis=dict(showgrid=True, zeroline=True),
                yaxis=dict(showgrid=True, zeroline=True),
                annotations=[dict(x=mean_residual, y=0, text="Mean: {:.2f}".format(mean_residual),
                                showarrow=True, arrowhead=2, ax=0, ay=-40)]
            )

            plot_html = fig.to_html(full_html=False)
            return render(request, 'analyze.html', {'plot_html': plot_html})
                
        elif action == 'show_ljung_box':
            # Show Ljung-Box Test results
            return render(request, 'analyze.html', {
                'ljung_box_results': ljung_box_results.to_dict(orient='records')  # Pass results as a list of dicts
            })

    return render(request, 'analyze.html', {'error': 'No data available for analysis!'})

def predict(request):
    # Load the trained model
    model = joblib.load('trained_arima_model.pkl')
    if request.method == 'POST':

        if request.POST.get('action') == 'download_csv':
            csv_data_json = request.session.get('forecast_data')
            if csv_data_json:
                forecast_df = pd.read_json(csv_data_json, convert_dates=['timestamp'])

                # Create the HTTP response with CSV content
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename=forecast.csv'
                forecast_df.to_csv(path_or_buf=response, index=False)
                return response

        # Get the number of steps for forecasting from the form
        try:
            forecast_steps = int(request.POST.get('steps'))
        except (ValueError, TypeError):
            return render(request, 'predict.html', {
                'error': 'Please enter a valid integer for forecast steps.'
            })

        # Load your dataset (assuming it's stored in the session as JSON)
        csv_data_json = request.session.get('csv_data')
        if csv_data_json:
            # Convert CSV data back to DataFrame
            csv_data = pd.read_json(csv_data_json, convert_dates=['timestamp'])

            # Sort the DataFrame by the timestamp to ensure chronological order
            csv_data = csv_data.sort_values(by='timestamp').reset_index(drop=True)

            train_size = int(len(csv_data) * 0.8)
            train_data = csv_data.iloc[:train_size+1]

            # Forecasting starting after the test data ends
            forecast = model.forecast(steps=forecast_steps)

            # Create a time index starting after the last test data timestamp
            last_timestamp = csv_data['timestamp'].iloc[train_size-1]
            forecast_index = pd.date_range(start=last_timestamp, periods=forecast_steps + 1, freq='D')[1:]

            # Create a DataFrame for the forecasted values
            forecast_df = pd.DataFrame({
                'timestamp': forecast_index,
                'value': forecast,
                'Type': 'Forecast'
            })
            
            # Combine the train, test, and forecast data into one DataFrame
            combined_data = pd.concat([
                train_data.assign(Type='Train'),
                forecast_df
            ]).sort_values(by='timestamp').reset_index(drop=True)

            # Save forecast data to the session for download later
            request.session['forecast_data'] = forecast_df.to_json()

            # Create the plot with Plotly
            fig = px.line(combined_data, x='timestamp', y='value', color='Type',
                          title='Forecast Data: Training and Prediction',
                          labels={'value': 'Value', 'timestamp': 'Timestamp', 'Type': 'Data Type'})
            
            # Add a range slider for better navigation
            fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=True)),
                yaxis_title='Value',
                legend_title='Data Type'
            )

            # Convert the Plotly figure to HTML
            plot_html = fig.to_html(full_html=False)

            # Render the template with the plot
            return render(request, 'predict.html', {'plot_html': plot_html})

    # If not POST, just show the form
    return render(request, 'predict.html')