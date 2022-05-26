from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error



def scatter_plot(data:pd.DataFrame, x_col, y_col,loss_metric = mean_squared_error, color_col = None,s=3,
                 x_label = "Predicted y",
                 y_label = "Actual y",
                 title = None):
    plt.style.use('seaborn-darkgrid')

    #todo check nan
    #Build a linear model
    corr_coef = scipy.stats.pearsonr(data[x_col], data[y_col])
    slope, intercept, r, p, stderr = scipy.stats.linregress(data[x_col], data[y_col])
    loss = loss_metric(data[y_col], data[x_col])
    mae = mean_absolute_error(data[y_col], data[x_col])
    #annotate MAE

    #plot points
    fig, ax = plt.subplots()
    if color_col is None:
        ax.scatter(x=data[x_col], y=data[y_col], s=s)
    else:
        colours = data[color_col].unique()
        #todo shoud test for integer/categorical
        for colour in colours:
            subset = data[data[color_col]==colour]
            ax.scatter(x=subset[x_col], y=subset[y_col], s=s, label=str(colour))

    #add linear model
    ax.plot(data[x_col], intercept + slope * data[x_col], label=f"y={slope:.4f}x+{intercept:.4f}", color="black")

    #add annotations
    ax.annotate(f"Pearsons correlation coefficient = {corr_coef[0]:.4f}", (0, 0), (0, -35), xycoords='axes fraction',
                textcoords='offset points', va='top', fontsize=12)
    ax.annotate(f"R^2 = {r ** 2:.4f}", (0, 0), (0, -55), xycoords='axes fraction', textcoords='offset points', va='top',
                fontsize=12)
    ax.annotate(f"MSE = {loss:.4f}", (0, 0), (0, -75), xycoords='axes fraction',
                textcoords='offset points', va='top', fontsize=12)
    ax.annotate(f"MAE = {mae:.4f}", (0, 0), (0, -95), xycoords='axes fraction',
                textcoords='offset points', va='top', fontsize=12)
    ax.legend(bbox_to_anchor=(1.04,1),loc='upper right')


    # plot residuals
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if not title is None:
        ax.set_title(title)

    #plt.savefig(log_dir + f"/models/{name}/predictions.png", bbox_inches='tight')
    #plt.close()

    return fig, ax

def residual_plot(data:pd.DataFrame, prediction, actual,loss_metric = mean_squared_error, color_col = None,s=3,
                 x_label = "Actual y",
                 y_label = "Actual y - Predicted y",
                 title = None ):
    plt.style.use('seaborn-whitegrid')

    residuals = data[actual]-data[prediction]

    #Build a linear model
    corr_coef = scipy.stats.pearsonr(data[actual],residuals)
    slope, intercept, r, p, stderr = scipy.stats.linregress(data[actual],residuals)
    loss = loss_metric(residuals,data[actual])
    mae = mean_absolute_error(residuals,data[actual])
    fig, ax = plt.subplots()
    if color_col is None:
        ax.scatter(x=data[actual], y=residuals, s=s)
    else:
        colours = data[color_col].unique()
        for colour in colours:
            subset = data[data[color_col] == colour]
            subset_residual = subset[actual]-subset[prediction]
            ax.scatter(x=subset[actual], y=subset_residual, s=s, label=str(colour))

    # add linear model
    ax.plot(data[actual], intercept + slope * data[actual], label=f"y={slope:.2f}x+{intercept:.2f}", color="black")

    # add annotations
    ax.annotate(f"Pearsons correlation coefficient = {corr_coef[0]:.4f}", (0, 0), (0, -35), xycoords='axes fraction',
                textcoords='offset points', va='top', fontsize=12)
    ax.annotate(f"R^2 = {r ** 2:.4f}", (0, 0), (0, -55), xycoords='axes fraction', textcoords='offset points', va='top',
                fontsize=12)
    ax.annotate(f"MSE = {loss:.4f}", (0, 0), (0, -75), xycoords='axes fraction',
                textcoords='offset points', va='top', fontsize=12)
    ax.annotate(f"MAE = {mae:.4f}", (0, 0), (0, -95), xycoords='axes fraction',
                textcoords='offset points', va='top', fontsize=12)
    ax.legend(bbox_to_anchor=(1.04,1),loc='upper right')

    # plot residuals
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if not title is None:
        ax.set_title(title)

    # plt.savefig(log_dir + f"/models/{name}/predictions.png", bbox_inches='tight')
    # plt.close()

    return fig, ax

def plot_preds_and_res(preds, save_loc="", name_lambda=lambda x: x, save_lambda=lambda x: x):
  for col_name in preds.columns:
    # plot predictions
    fig, ax = scatter_plot(preds, col_name, "y", color_col="set_id",
                                title=f"Predictions for {name_lambda(col_name)}")
    plt.savefig(save_loc / f"predictions_{save_lambda(col_name)}.png", bbox_inches='tight')
    plt.close()
    # plt.show()

    fig, ax = residual_plot(preds, col_name, "y", color_col="set_id",
                                 title=f"Residuals for {name_lambda(col_name)}")
    plt.savefig(save_loc / f"residuals_{save_lambda(col_name)}.png", bbox_inches='tight')
    plt.close()
