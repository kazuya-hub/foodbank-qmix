import subprocess


def main():

    for i in range(100):

        for a in reversed([20]):
            for f in reversed([20]):
                eplimit = f * 5
                experiment_command = f"""python3 src/main.py --algo qmix --env foodbank --situ "{a}a{f}f_lc1" --tmax "1000000" --eplimit "{eplimit}" --wandb"""
                print(experiment_command)
                subprocess.call(experiment_command, shell=True)


if __name__ == "__main__":
    main()
