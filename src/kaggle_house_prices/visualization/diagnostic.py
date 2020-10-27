from matplotlib import pyplot as plt


class PredictionPlot:
    def __init__(self,
                 title="Predicted vs Actual",
                 training_colour="blue", test_colour="lightgreen"):
        self.training_colour = training_colour
        self.test_colour = test_colour
        self.plot_title = title

        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self, training_predictions, test_predictions, training_actual, test_actual):
        self.ax.scatter(training_predictions,
                        training_actual,
                        c=self.training_colour,
                        marker="s",
                        label="Training data")
        self.ax.scatter(test_predictions,
                        test_actual,
                        c=self.test_colour,
                        marker="s",
                        label="Validation")
        self.ax.set_title(self.plot_title)
        self.ax.set_xlabel("Predicted Values")
        self.ax.set_ylabel("Real Values")
        self.ax.legend(loc="upper left")
        self.ax.plot([10.5, 13.5], [10.5, 13.5], c="red")

        return self.ax

    def save(self, file_path):
        self.fig.savefig(file_path)


def plot_residuals(training_predictions, test_predictions, training_actual, test_actual):
    fig, ax = plt.subplots(1, 1)
    plt.scatter(training_predictions,
                training_predictions - training_actual,
                c="blue",
                marker="s",
                label="Training data")
    plt.scatter(test_predictions,
                test_predictions - test_actual,
                c="lightgreen",
                marker="s",
                label="Validation data")
    plt.title("Linear Regression")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")