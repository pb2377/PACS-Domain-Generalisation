from utils import Visualiser


def main():
    visualiser = Visualiser()
    logdir = 'save/resnet18/pacs/photo-0/'
    visualiser(logdir=logdir)


if __name__ == '__main__':
    main()
