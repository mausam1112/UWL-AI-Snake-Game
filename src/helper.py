import os 
import matplotlib.pyplot as plt

from IPython import display


plt.ion()

def plot(scores, mean_scores, model_folder_path):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.plot(range(1, len(scores)+1), scores, label="Score")
    plt.plot(range(1, len(mean_scores)+1), mean_scores, label="Mean Score")

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
    plt.savefig(f"{model_folder_path}/score.png")


def get_model_suffix(folder="./models"):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return 1

    max_suffix = 0

    for f in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, f)):
            # Extracting the suffix after last '-'
            suffix = f.rsplit("-", 1)[-1]

            # when possible, convert suffix to int
            try:
                num = int(suffix)
                max_suffix = max(max_suffix, num)
            except ValueError:
                pass  # ignore files without numeric suffix

    return (max_suffix if max_suffix > 0 else 0) + 1 # adding 1 for new suffix


if __name__ == "__main__":
    print(get_model_suffix())