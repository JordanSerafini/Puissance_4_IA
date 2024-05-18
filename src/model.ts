import * as tf from '@tensorflow/tfjs';

const numRows = 6;
const numCols = 7;

export const createModel = () => {
  const model = tf.sequential();

  // Ajout de couches au modèle
  model.add(tf.layers.dense({ inputShape: [numCols * numRows], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: numCols, activation: 'softmax' }));

  // Compiler le modèle
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};
