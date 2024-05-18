import { createModel } from './model';
import { generateTrainingData } from './data';
import * as tf from '@tensorflow/tfjs';

const player1Model = createModel();
const player2Model = createModel();

const trainModel = async (model: tf.Sequential, xs: tf.Tensor, ys: tf.Tensor) => {
  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch: number, logs?: tf.Logs) => {
        if (logs) {
          console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
        }
      },
    },
  });
};

const train = async () => {
  for (let i = 0; i < 10; i++) {
    const { xs, ys } = generateTrainingData(player1Model, player2Model);
    await trainModel(player1Model, xs, ys);
    await trainModel(player2Model, xs, ys);
  }
};

train().then(() => {
  console.log('Models trained');
});
