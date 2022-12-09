# most of here comes from https://juliaai.github.io/DataScienceTutorials.jl/end-to-end
using MLJ, MLJFlux, MLJScikitLearnInterface, PrettyPrinting, CSV
import DataFrames: DataFrame, select!, Not, describe, rename!
import Statistics
using Plots, Plots.Measures
default(size=(900,400),fontfamily="Computer Modern", linewidth=4, framestyle=:box, margin=5mm); scalefontsizes(); scalefontsizes(1.3)

"""
    plot_confusion_matrix(y_pred,y_test; normalise=false)

Display the confusion matrix for predicted `y_pred` and true `y_test` values without normalisation as default.
"""
function plot_confusion_matrix(y_pred,y_test; normalise=false)
    cm = confusion_matrix(mode.(y_pred), y_test).mat
    normalise && (cm = 100*cm./sum(cm,dims=2))

    plt = heatmap(cm, c=:turbo, xlabel="True", ylabel="Predicted", title="confusion matrix",
        xticks=([1,2], ["no landslide", "landlside"]),
        yticks=([1,2], ["no landslide", "landlside"]),yrotation=90)

    for i ∈ axes(cm,1), j ∈ axes(cm,2)
        annotate!(i,j, string(round(Int,cm[i,j]))*"%", :white)
    end
    return plt
end


"""
    prepare_data(input_file; show_info=true)

Prepare the data and split it into train and test set.
Return `X, y, train, test, df, df2`
"""
function prepare_data(input_file; show_info=true)

    df = CSV.File(input_file) |> DataFrame

    # Fix types
    for n in names(df)
        if n == "LandCover" || n == "Geology" || n == "UID"
            df[!,n] .= convert.(Union{Int64, Missing},df[!,n])
        else
            df[!,n] .= convert.(Union{Float64, Missing},df[!,n])
        end
    end

    # Change categorical data into ordered factor
    # coerce!(df, :Geology => OrderedFactor, :LandCover => OrderedFactor)
    coerce!(df, :Geology => Multiclass, :LandCover => Multiclass)

    # one hot encode ordered factors
    hot = OneHotEncoder()
    mach = machine(hot, df) |> fit!
    df2 = MLJ.transform(mach, df)

    coerce!(df2, :LS => OrderedFactor{2})

    # separate the target variable y from the feature set X
    y, X = unpack(df2, ==(:LS))
    # y, dst_rd, X = unpack(df2, ==(:LS), ==(:dist_roads)) # remove distance to roads

    # Standardise
    transformer_instance = Standardizer()
    transformer_model = machine(transformer_instance, X) |> fit!
    X = MLJ.transform(transformer_model, X)

    # Train-test split
    train, test = partition(collect(eachindex(y)), 0.7, shuffle=true, rng=5)

    # Retrieve applicable models
    show_info==true && for m in models(matching(X, y))
        println("Model name = ",m.name,", ","Prediction type = ",m.prediction_type,", ","Package name = ",m.package_name);
    end
    return X, y, train, test, df, df2
end
####################################################################

run        = :single
input_file = "data/Landslides.csv"
save_fig   = false
save_mach  = (run==:single) ? true : false
mach_name  = "my_machine_tmp.jlso"

X, y, train, test, df, df2 = prepare_data(input_file; show_info=true)

if run == :multi
    model_names=Vector{String}(); loss_acc=[]; loss_ce=[]; loss_f1=[]; p3=plot()
    for m in models(matching(X, y))
        if m.prediction_type==Symbol("probabilistic") && (m.package_name=="MLJFlux" || m.package_name=="ScikitLearn") && 
            # m.name=="DecisionTreeClassifier"
            m.name!="LogisticCVClassifier" && m.name!="GaussianProcessClassifier" && m.name!="KNeighborsClassifier" && m.name!="DummyClassifier"
            # Excluding LogisticCVClassfiier as we can infer similar baseline results from the LogisticClassifier

            # Capturing the model and loading it using the @load utility
            model_name   = m.name
            package_name = m.package_name
            eval(:(clf = @load $model_name pkg=$package_name verbosity=1))

            # Fitting the captured model onto the training set
            clf_machine = machine(clf(), X, y)
            fit!(clf_machine, rows=train)

            # Getting the predictions onto the test set
            y_pred = MLJ.predict(clf_machine, rows=test)

            # Plotting the ROC-AUC curve for each model being iterated
            fprs, tprs, thresholds = roc(y_pred, y[test])
            display(plot!(fprs, tprs,label=model_name))

            # Obtaining different evaluation metrics
            ce_loss  = mean(cross_entropy(y_pred,y[test]))
            acc      = accuracy(mode.(y_pred), y[test])
            f1_score = f1score(mode.(y_pred), y[test])

            # Adding values of the evaluation metrics to the respective vectors
            push!(model_names, m.name)
            append!(loss_acc, acc)
            append!(loss_ce, ce_loss)
            append!(loss_f1, f1_score)
        end
    end

    # Labels and legend for the ROC-AUC curve
    display(plot!(title="ROC curve",xlabel="False Positive Rate",ylabel="True Positive Rate"))
    save_fig && png(plot(p3,dpi=300), "docs/compare.png")
    df_out = DataFrame(M=model_names, A=loss_acc, B=loss_ce, C=loss_f1)
    rename!(df_out, :M => :Model, :A => :Accuracy, :B => :Mean_cross_entropy, :C => :f1_scores)
    sort!(df_out,:Accuracy,rev=true)
    display(df_out)

elseif run == :single

    # Selecting one model
    # clf = @load GradientBoostingClassifier pkg=ScikitLearn verbosity=2
    # clf = @load RandomForestClassifier pkg=ScikitLearn verbosity=0
    clf = @load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

    # Init the model
    # CLF = clf()
    # Hand-tune the model
    CLF = clf(epochs=600,
              batch_size = 32,
              lambda = 0.05,
              alpha = 0.001,
              acceleration=CPUThreads())
              #acceleration=CUDALibs())
    CLF.optimiser.eta = 0.001
    
    # Train the model
    clf_machine = machine(CLF, X, y)
    fit!(clf_machine, rows=train, verbosity=1)

    # Predict
    y_train = MLJ.predict(clf_machine, rows=train)
    fprs_i, tprs_i, thresholds = roc(y_train, y[train])

    # Save machine for further application
    save_mach && MLJ.save(mach_name, clf_machine)
    
    # Validate
    y_pred = MLJ.predict(clf_machine, rows=test)
    fprs, tprs, thresholds = roc(y_pred, y[test])

    # Plotting the ROC-AUC curve for model
    p1 = plot(fprs_i, tprs_i, label="train")
    display(plot!(fprs, tprs, label="validate"))
    plot!(title="ROC curve",xlabel="False Positive Rate",ylabel="True Positive Rate")
    p2 = plot_confusion_matrix(y_pred,y[test]; normalise=true)
    display(plot(p1,p2))
    save_fig && png(plot(p1,p2,dpi=300), "docs/roc_cm.png")

    # Obtaining different evaluation metrics
    println("Model evaluation metrics")
    println("- cross entropy loss: $(mean(cross_entropy(y_pred,y[test])))")
    println("- accuracy: $(accuracy(mode.(y_pred), y[test]))")
    println("- f1 score: $(f1score(mode.(y_pred), y[test]))")
else   
    @warn "Run type not defined"
end
