from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect, send_from_directory, send_file
from wtforms.validators import DataRequired, Optional
from wtforms import SubmitField, FileField, IntegerField, TextAreaField, FloatField
from ensembles import RandomForestMSE, GradientBoostingMSE
import pandas as pd
import os
import re
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import time


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'prod'
Bootstrap(app)
rf_result_directory = './server/flaskr/rf_results'
gbm_result_directory = './server/flaskr/gbm_results'
rf_dataset_directory = './server/flaskr/rf_datasets'
gbm_dataset_directory = './server/flaskr/gbm_datasets'
rf_info_directory = './server/flaskr/static/rf_info'
gbm_info_directory = './server/flaskr/static/gbm_info'


class RFMakeForm(FlaskForm):
    n_estimators = IntegerField('Number of trees.', validators=[DataRequired()])
    max_depth = IntegerField('Max depth. Unlimited by default.', validators=[Optional()])
    feature_subsample_size = IntegerField('Feature subsample size. Uses all features by default.',
                                          validators=[Optional()])
    other_params = TextAreaField('Other trees parameters in format: parameter_name=parameter_value. '
                                 'Default values like in sklearn.tree.DecisionTreeRegressor.',
                                 validators=[Optional()])
    submit = SubmitField('Create model')


class GBMMakeForm(FlaskForm):
    n_estimators = IntegerField('Number of trees.', validators=[DataRequired()])
    learning_rate = FloatField('Learning rate. Default is 0.1.', validators=[Optional()])
    max_depth = IntegerField('Max depth. Default is 5.', validators=[Optional()])
    feature_subsample_size = IntegerField('Feature subsample size. Uses all features by default.',
                                          validators=[Optional()])
    other_params = TextAreaField("""Other trees parameters in format: parameter_name=parameter_value.
                                 Default values like in sklearn.tree.DecisionTreeRegressor.""",
                                 validators=[Optional()])
    submit = SubmitField('Create model')


class ModelParametersForm(FlaskForm):
    submit1 = SubmitField('View model parameters')


class ModelDatasetXForm(FlaskForm):
    submit1 = SubmitField('Download X_train')


class ModelDatasetYForm(FlaskForm):
    submit1 = SubmitField('Download y_train')


class ModelTrainingInfo(FlaskForm):
    submit1 = SubmitField('View training process info')


class TrainModelForm(FlaskForm):
    file_path_data = FileField('Path to .csv data file', validators=[DataRequired()])
    file_path_target = FileField('Path to .csv target file', validators=[DataRequired()])
    submit1 = SubmitField('Train model')


class PredictForm(FlaskForm):
    file_path = FileField('Path to .csv data file', validators=[DataRequired()])
    submit1 = SubmitField('Get predictions')


class GetResForm(FlaskForm):
    submit1 = SubmitField('Download predictions')


rf_model = None
gbm_model = None


@app.route('/make_rf', methods=['GET', 'POST'])
def make_rf():
    rf_form = RFMakeForm()

    if request.method == 'POST' and rf_form.validate_on_submit():
        global rf_model
        other_params_list = re.split("[^a-zA-Z0-9='._]+", rf_form.other_params.data)
        other_params_dict = {}
        if len(other_params_list) > 1 or other_params_list[0]:
            for pair in other_params_list:
                other_params_dict[pair.split('=')[0]] = eval(pair.split('=')[1])
        rf_model = RandomForestMSE(n_estimators=rf_form.n_estimators.data, max_depth=rf_form.max_depth.data,
                                   feature_subsample_size=rf_form.feature_subsample_size.data, **other_params_dict)
        return redirect(url_for('rf', fitted=False, res=False))

    return render_template('from_form1_rf.html', form=rf_form)


@app.route('/make_gbm', methods=['GET', 'POST'])
def make_gbm():
    gbm_form = GBMMakeForm()

    if request.method == 'POST' and gbm_form.validate_on_submit():
        global gbm_model
        other_params_list = re.split("[^a-zA-Z0-9='._]+", gbm_form.other_params.data)
        other_params_dict = {}
        if len(other_params_list) > 1 or other_params_list[0]:
            for pair in other_params_list:
                other_params_dict[pair.split('=')[0]] = eval(pair.split('=')[1])
        learning_rate = 0.1 if gbm_form.learning_rate.data is None else gbm_form.learning_rate.data
        max_depth = 5 if gbm_form.max_depth.data is None else gbm_form.max_depth.data
        gbm_model = GradientBoostingMSE(n_estimators=gbm_form.n_estimators.data, learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        feature_subsample_size=gbm_form.feature_subsample_size.data,
                                        **other_params_dict)
        return redirect(url_for('gbm', fitted=False, res=False))

    return render_template('from_form1_gbm.html', form=gbm_form)


@app.route('/rf', methods=['GET', 'POST'])
def rf():
    params_form = ModelParametersForm()
    train_form = TrainModelForm()
    pred_form = PredictForm()
    res_form = GetResForm()
    datasetx_form = ModelDatasetXForm()
    datasety_form = ModelDatasetYForm()
    info_form = ModelTrainingInfo()

    if request.method == 'POST' and params_form.validate_on_submit() and \
            request.form['submit1'] == 'View model parameters':
        return redirect(url_for('rf_params'))
    if request.method == 'POST' and datasetx_form.validate_on_submit() and \
            request.form['submit1'] == 'Download X_train':
        return send_file('./rf_datasets/data.csv', as_attachment=True)
    if request.method == 'POST' and datasety_form.validate_on_submit() and \
            request.form['submit1'] == 'Download y_train':
        return send_file('./rf_datasets/target.csv', as_attachment=True)
    if request.method == 'POST' and info_form.validate_on_submit() and \
            request.form['submit1'] == 'View training process info':
        return redirect(url_for('rf_info'))
    if request.method == 'POST' and train_form.validate_on_submit() and request.form['submit1'] == 'Train model':
        data = pd.read_csv(train_form.file_path_data.data, index_col='index')
        target = pd.read_csv(train_form.file_path_target.data, index_col='index')
        data.to_csv(os.path.join(rf_dataset_directory, 'data.csv'))
        target.to_csv(os.path.join(rf_dataset_directory, 'target.csv'))
        rf_model.fit(data.values, target.values.ravel())
        return redirect(url_for('rf', fitted=True, res=False))
    if request.method == 'POST' and pred_form.validate_on_submit() and request.form['submit1'] == 'Get predictions':
        if rf_model.fitted:
            data = pd.read_csv(pred_form.file_path.data, index_col='index')
            res = rf_model.predict(data.values)
            res = pd.DataFrame(res, columns=['predictions'], index=data.index)
            res.to_csv(os.path.join(rf_result_directory, 'res.csv'))
            return redirect(url_for('rf', fitted=True, res=True))
        else:
            return redirect(url_for('rf', fitted=False, res=False))
    if request.method == 'POST' and res_form.validate_on_submit() and request.form['submit1'] == 'Download predictions':
        return send_file('./rf_results/res.csv', as_attachment=True)

    if request.method == 'GET' and request.args['fitted'] == 'False':
        return render_template('from_form2.html', params_form=params_form, train_form=train_form, pred_form=pred_form)
    if request.method == 'GET' and request.args['fitted'] == 'True' and request.args['res'] == 'False':
        return render_template('from_form3.html', params_form=params_form, train_form=train_form, pred_form=pred_form,
                               datasetx_form=datasetx_form, datasety_form=datasety_form, info_form=info_form)
    if request.method == 'GET' and request.args['fitted'] == 'True' and request.args['res'] == 'True':
        return render_template('from_form4.html', params_form=params_form, train_form=train_form, pred_form=pred_form,
                               res_form=res_form, datasetx_form=datasetx_form, datasety_form=datasety_form,
                               info_form=info_form)


@app.route('/gbm', methods=['GET', 'POST'])
def gbm():
    params_form = ModelParametersForm()
    train_form = TrainModelForm()
    pred_form = PredictForm()
    res_form = GetResForm()
    datasetx_form = ModelDatasetXForm()
    datasety_form = ModelDatasetYForm()
    info_form = ModelTrainingInfo()

    if request.method == 'POST' and params_form.validate_on_submit() and \
            request.form['submit1'] == 'View model parameters':
        return redirect(url_for('gbm_params'))
    if request.method == 'POST' and datasetx_form.validate_on_submit() and \
            request.form['submit1'] == 'Download X_train':
        return send_file('./gbm_datasets/data.csv', as_attachment=True)
    if request.method == 'POST' and datasety_form.validate_on_submit() and \
            request.form['submit1'] == 'Download y_train':
        return send_file('./gbm_datasets/target.csv', as_attachment=True)
    if request.method == 'POST' and info_form.validate_on_submit() and \
            request.form['submit1'] == 'View training process info':
        return redirect(url_for('gbm_info'))
    if request.method == 'POST' and train_form.validate_on_submit() and request.form['submit1'] == 'Train model':
        data = pd.read_csv(train_form.file_path_data.data, index_col='index')
        target = pd.read_csv(train_form.file_path_target.data, index_col='index')
        data.to_csv(os.path.join(gbm_dataset_directory, 'data.csv'))
        target.to_csv(os.path.join(gbm_dataset_directory, 'target.csv'))
        gbm_model.fit(data.values, target.values.ravel())
        return redirect(url_for('gbm', fitted=True, res=False))
    if request.method == 'POST' and pred_form.validate_on_submit() and request.form['submit1'] == 'Get predictions':
        if gbm_model.fitted:
            data = pd.read_csv(pred_form.file_path.data, index_col='index')
            res = gbm_model.predict(data.values)
            res = pd.DataFrame(res, columns=['predictions'], index=data.index)
            res.to_csv(os.path.join(gbm_result_directory, 'res.csv'))
            return redirect(url_for('gbm', fitted=True, res=True))
        else:
            return redirect(url_for('gbm', fitted=False, res=False))
    if request.method == 'POST' and res_form.validate_on_submit() and request.form['submit1'] == 'Download predictions':
        return send_file('./gbm_results/res.csv', as_attachment=True)

    if request.method == 'GET' and request.args['fitted'] == 'False':
        return render_template('from_form2_gbm.html', params_form=params_form, train_form=train_form,
                               pred_form=pred_form)
    if request.method == 'GET' and request.args['fitted'] == 'True' and request.args['res'] == 'False':
        return render_template('from_form3_gbm.html', params_form=params_form, train_form=train_form,
                               pred_form=pred_form, datasetx_form=datasetx_form, datasety_form=datasety_form,
                               info_form=info_form)
    if request.method == 'GET' and request.args['fitted'] == 'True' and request.args['res'] == 'True':
        return render_template('from_form4_gbm.html', params_form=params_form, train_form=train_form,
                               pred_form=pred_form, res_form=res_form, datasetx_form=datasetx_form,
                               datasety_form=datasety_form, info_form=info_form)


@app.route('/rf_params', methods=['GET'])
def rf_params():
    params = {'n_estimators': rf_model.n_estimators, 'max_depth': rf_model.max_depth,
              'max_features': rf_model.feature_subsample_size, **rf_model.trees_parameters}
    params_default = DecisionTreeRegressor().get_params()
    params_default.update(params)
    return render_template('rf_params.html', params=params_default)


@app.route('/gbm_params', methods=['GET'])
def gbm_params():
    params = {'n_estimators': gbm_model.n_estimators, 'max_depth': gbm_model.max_depth,
              'max_features': gbm_model.feature_subsample_size, 'learning_rate': gbm_model.learning_rate,
              **gbm_model.trees_parameters}
    params_default = DecisionTreeRegressor().get_params()
    params_default.update(params)
    return render_template('gbm_params.html', params=params_default)


@app.route('/rf_info', methods=['GET'])
def rf_info():
    fig = plt.figure(figsize=(18, 10))
    plt.plot(np.arange(1, rf_model.n_estimators + 1), rf_model.loss_func, color='red')
    plt.title('Loss function (MSE) during training', fontsize=20)
    plt.xlabel('Iteration number', fontsize=15)
    plt.ylabel('Loss function value', fontsize=15)
    plt.grid()
    fig.savefig(os.path.join(rf_info_directory, 'loss.jpg'), dpi=100)
    return render_template('rf_info.html', rand=str(time.time()))


@app.route('/gbm_info', methods=['GET'])
def gbm_info():
    fig = plt.figure(figsize=(18, 10))
    plt.plot(np.arange(1, gbm_model.n_estimators + 1), gbm_model.loss_func, color='red')
    plt.title('Loss function (MSE) during training', fontsize=20)
    plt.xlabel('Iteration number', fontsize=15)
    plt.ylabel('Loss function value', fontsize=15)
    plt.grid()
    fig.savefig(os.path.join(gbm_info_directory, 'loss.jpg'), dpi=100)
    return render_template('gbm_info.html', rand=str(time.time()))


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/index_js')
def get_index():
    return '<html><center><script>document.write("Hello, i`am Flask Server!")</script></center></html>'
