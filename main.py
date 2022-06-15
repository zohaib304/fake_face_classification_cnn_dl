""" main.py """

from configs.config import CFG
from model.cnn import CNN


def run():
    """ Build model, loads data, trains"""
    model = CNN(CFG)
    # mymodel.load_data()
    model.build()
    model.train()
    model.evaluate()
    model.predicting()
    model.saving()


if __name__ == '__main__':
    run()
