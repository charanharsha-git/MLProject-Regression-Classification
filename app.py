import pandas as pd
from flask import Flask, render_template, request
import model as m
import model2 as m2
app = Flask(__name__)

df=None
col_list=None
dependent_var=None
ret=None

@app.route('/',methods=["POST","GET"])
def select_reg_class():
    global ret
    if request.method == "POST":



        return render_template("main.html")
    return render_template("main.html")

@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    return render_template(str(select)+".html")

@app.route('/Regression',methods=["POST","GET"])
def upload():
    global df
    global col_list
    if request.method == "POST":

        df = pd.read_csv(request.files.get('myfile'))
        col_list = df.columns
        return render_template("Regression.html",data_frame=df,column_list=col_list)
    return render_template("Regression.html")





@app.route('/SelectVariable',methods=["POST","GET"])
def select_variable():
    upload()
    global dependent_var
    global dep_var
    if request.method == "POST":

        def dep_var():
            ii=1
            #dependent_var = request.form["dname"]
        return render_template("select_variable.html", data_frame=df,dependent_var=dependent_var, column_list=col_list)

    return render_template("select_variable.html")

@app.route('/ModelResults',methods=["POST","GET"])
def model_results():

    if request.method == "POST":

        #dep_var()
        df = pd.read_csv(request.files.get('myfile'))
        dependent_var = request.form["dname"]
        preprocessed=m.preprocessing(df)
        X,Y=m.x_n_y(preprocessed,dependent_var)
        x_train, x_test, y_train, y_test=m.train_test_split1(X,Y)
        rmse= m.models(x_train,y_train,x_test,y_test)

        return render_template("model results.html",scores=rmse,data_frame=df,dependent_var=dependent_var)
    return render_template("model results.html")

@app.route('/Classification',methods=["POST","GET"])
def upload2():
    global df
    global col_list
    if request.method == "POST":

        df = pd.read_csv(request.files.get('myfile'))
        col_list = m2.cat_var(df)
        print(col_list)
        return render_template("Classification.html",data_frame=df,column_list=col_list)
    return render_template("Classification.html")

@app.route('/SelectVariable2',methods=["POST","GET"])
def select_variable2():
    upload2()
    global dependent_var
    global dep_var
    if request.method == "POST":

        def dep_var():
            ii=1
            #dependent_var = request.form["dname"]
        return render_template("select_variable2.html", data_frame=df,dependent_var=dependent_var, column_list=col_list)

    return render_template("select_variable2.html")

@app.route('/ModelResults2',methods=["POST","GET"])
def model_results2():

    if request.method == "POST":

        #dep_var()
        dependent_var = request.form["dname"]
        preprocessed=m2.preprocessing(df)
        X,Y=m2.x_n_y(preprocessed,dependent_var)
        x_train, x_test, y_train, y_test=m2.train_test_split1(X,Y)
        accuracies,classifiers= m2.models(x_train,y_train,x_test,y_test)

        return render_template("model results2.html",scores=accuracies,data_frame=df,dependent_var=dependent_var,classifiers=classifiers)
    return render_template("model results2.html")

@app.route('/Classification plot',methods=["POST","GET"])
def classification_plot():
    if request.method == "POST":
        oo=1
        return render_template("classification_plot.html")
    return render_template("classification_plot.html")

@app.route('/Regression plot',methods=["POST","GET"])
def regression_plot():
    if request.method == "POST":
        kk=1
        return render_template("regression_plot.html")
    return render_template("regression_plot.html")


if __name__=='__main__':
    app.run(debug=True)
