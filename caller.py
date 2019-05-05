import os

if __name__ == "__main__":
    folder = "Bi-AttnLocation"
    if not os.path.exists(folder):
        os.mkdir(folder)
    for i in range(30):
        output_path = os.path.join(folder, "{}.csv".format(i))
        print(output_path)
        os.system("python main.py >> {}".format(output_path))