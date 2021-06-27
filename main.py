""" main.py """

from configs.config import CFG
from model.cnn import CNN


def run():
    """ Build model, loads data, trains"""
    mymodel = CNN(CFG)
    # mymodel.load_data()
    mymodel.build()
    mymodel.train()
    mymodel.evaluate()
    mymodel.predicting()
    mymodel.saving()


if __name__ == '__main__':
    run()
