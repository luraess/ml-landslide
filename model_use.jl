# misc
using Rasters, DataFrames
using MLJ, PrettyPrinting
import Statistics
using Plots, Plots.Measures
default(size=(900,400),fontfamily="Computer Modern", linewidth=4, framestyle=:box, margin=5mm); scalefontsizes(); scalefontsizes(1.3)

"""
    vis_classes(df_all, data, c; vis_class=true)

Visualise results from field `data` extracted from DataFrame `df_all` in `c` classes.
"""
function vis_classes(df_all, data, c; vis_class=true)
    # Create bins for classes
    ls = df_all[!,data]
    ls2 = copy(ls); ls2[ismissing.(ls2)].=NaN
    ls_class = zeros(size(ls))
    ls_class[ ls2 .>= c[5]] .= 6
    ls_class[(ls2 .< c[5]) .& (ls2 .>= c[4])] .= 5
    ls_class[(ls2 .< c[4]) .& (ls2 .>= c[3])] .= 4
    ls_class[(ls2 .< c[3]) .& (ls2 .>= c[2])] .= 3
    ls_class[(ls2 .< c[2]) .& (ls2 .>= c[1])] .= 2
    ls_class[(ls2 .< c[1]) .& (ls2 .>= 0)   ] .= 1
    ls_class[ismissing.(ls)].=NaN

    xc = df_all[!,:X]
    yc = df_all[!,:Y]
    x, y = minimum(xc):25:maximum(xc), minimum(yc):25:maximum(yc)
    nx, ny = length(x), length(y)
    @assert nx*ny == length(xc) # make sure we do not produce entropy
    
    tp = (vis_class==true) ? ls_class : ls
    ls = reverse(reshape(tp,nx,ny),dims=2)'

    default(size=(600,500))
    plt=heatmap(x,y,ls, color=reverse(palette(:inferno,6)), aspect_ratio=:equal,
                xlim=extrema(x), ylim=extrema(y), grid = false, cbar=true,
                xlabel="x-coord", ylabel="y-coord", title="Prob. landslide occurance %")
    return plt
end

"""
    prepare_data(d; checkkey=false)

Prepare the data based on Dict `d` entries (check entries set `checkkey=true`).
Return `X, uid, df, dfm`, the feature data, a UID, a Dict with and without missing val.
"""
function prepare_data(d; checkkey=false)
    if checkkey
        for (key, value) in d
            print(key); println(" | " * value)
        end
    end
    df = Raster(get(d,"DEM",missing)) |> replace_missing |> DataFrame
    select!(df, Not(size(df,2)))
    select!(df, Not(:Band))

    for (key, value) in d
        insertcols!(df, size(df,2)+1, key => DataFrame(Raster(value) |> replace_missing)[:,end])
    end

    # Create a UUID
    insertcols!(df, 1, "UID" => 1:size(df,1))

    # Fix types
    for n in names(df)
        if n == "LandCover" || n == "Geology" || n == "UID"
            df[!,n] .= convert.(Union{Int64, Missing},df[!,n])
        else
            df[!,n] .= convert.(Union{Float64, Missing},df[!,n])
        end
    end

    # Drop missing data
    dfm = dropmissing(df)
    # coerce!(dfm, :Geology => OrderedFactor, :LandCover => OrderedFactor)
    coerce!(dfm, :Geology => Multiclass, :LandCover => Multiclass)

    hot = OneHotEncoder()
    mach = machine(hot, dfm) |> fit!
    df2 = MLJ.transform(mach, dfm)

    # Exclude unwanted features from data-set
    # NOTE: also exclude geol_37 as not present in training !!!
    xc, yc, uid, geol_37, X = unpack(df2, ==(:X), ==(:Y), ==(:UID), ==(:Geology__37))
    # xc, yc, uid, geol_37, dst_rd, X = unpack(df2, ==(:X), ==(:Y), ==(:UID), ==(:Geology__37), ==(:dist_roads)) # remove distance to roads

    # Standardise
    transformer_instance = Standardizer()
    transformer_model = machine(transformer_instance, X) |> fit!
    X = MLJ.transform(transformer_model, X)

    return X, uid, df, dfm
end
####################################################################

d = Dict( "DEM" => "data/DEM.tif",
          "Geology" => "data/Geology.tif",
          "dist_roads" => "data/dist_roads.tif",
          "LandCover" => "data/LandCover.tif",
          "plan_curvature" => "data/plan_curvature.tif",
          "profil_curvature" => "data/profil_curvature.tif",
          "Slope" => "data/Slope.tif",
          "TWI" => "data/TWI.tif" )

save_fig  = true
mach_name = "my_machine_tmp.jlso"

# Load trained machine
# @load GradientBoostingClassifier pkg=ScikitLearn verbosity=2
# @load RandomForestClassifier pkg=ScikitLearn verbosity=0
@load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

trained_mach = machine(mach_name)

# Prepare data
X, uid, df, dfm = prepare_data(d)

# Predict
ls_pred = MLJ.predict(trained_mach, X)

# Retrieve probability of landslide occurrence 
LS = ls_pred.prob_given_ref[2]
uid = Int.(uid)

# Create a new DataFrame with output
df_LS = DataFrame(hcat(uid,LS),[:UID, :LS])
df_LS[!,:UID] .= convert.(Int64,df_LS[!,:UID])

# Get x,y coords back for output
df_all = sort(leftjoin(df, df_LS, on="UID"), :UID)
df_all[!,:X] .= convert.(Int64,round.(Int,df_all[!,:X]))
df_all[!,:Y] .= convert.(Int64,round.(Int,df_all[!,:Y]))

# Define classes of probabilities
c = (0.25, 0.5, 0.6, 0.7, 0.8)

# Visu
plt = vis_classes(df_all, :LS, c; vis_class=true)
display(plot(plt))
save_fig && png(plot(plt,dpi=300), "docs/ls_vd.png");
