from trainer import *
from predictor import *
def train():
    args = get_args()
    trainer = Trainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('Exiting from training early')

def predict():
    args = get_args()
    args.snapshot = './snapshot/2018-06-25_17-41-30/best_steps_4800.pt'
    predictor = Predictor(args)
    with open('test.txt' ,'r' ,encoding='utf-8') as f:
        for line in f.readlines():
            print(predictor.predict(line), end=' ')

if __name__ == '__main__':
    train()
    #predict()