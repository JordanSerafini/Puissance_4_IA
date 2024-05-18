import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

// Types pour les joueurs
type Player = 'R' | 'Y';

// Constantes pour les dimensions de la grille
const numRows = 6;
const numCols = 7;

// Initialisation de la matrice de jeu
const initializeMatrix = (): Array<Array<string | null>> => {
  return Array(numCols).fill(null).map(() => Array(numRows).fill(null));
};

// Ajouter un jeton dans une colonne
const addTokenToCol = (matrix: Array<Array<string | null>>, col: number, player: Player): Array<Array<string | null>> => {
  if (col < 0 || col >= numCols) {
    throw new Error('Out of range');
  }

  for (let row = numRows - 1; row >= 0; row--) {
    if (matrix[col][row] === null) {
      matrix[col][row] = player;
      return matrix;
    }
  }

  throw new Error('Column is full');
};

// Vérifier la direction pour la victoire
const checkDirection = (matrix: Array<Array<string | null>>, startRow: number, startCol: number, deltaRow: number, deltaCol: number, player: Player): boolean => {
  let count = 0;
  for (let i = 0; i < 4; i++) {
    const row = startRow + i * deltaRow;
    const col = startCol + i * deltaCol;
    if (row >= 0 && row < numRows && col >= 0 && col < numCols && matrix[col][row] === player) {
      count++;
    } else {
      break;
    }
  }
  return count === 4;
};

// Vérifier la victoire
const checkWin = (matrix: Array<Array<string | null>>, player: Player): boolean => {
  for (let col = 0; col < numCols; col++) {
    for (let row = 0; row < numRows; row++) {
      if (
        checkDirection(matrix, row, col, 0, 1, player) || 
        checkDirection(matrix, row, col, 1, 0, player) || 
        checkDirection(matrix, row, col, 1, 1, player) || 
        checkDirection(matrix, row, col, 1, -1, player)   
      ) {
        return true;
      }
    }
  }
  return false;
};

// Définir le modèle
const createModel = () => {
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

const model = createModel();

const generateTrainingData = () => {
  const inputs: number[][] = [];
  const outputs: number[][] = [];

  for (let i = 0; i < 1000; i++) {
    const board = Array(numCols * numRows).fill(0);
    const move = Math.floor(Math.random() * numCols);
    inputs.push(board);
    const output = Array(numCols).fill(0);
    output[move] = 1;
    outputs.push(output);
  }

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(outputs);

  return { xs, ys };
};

const { xs, ys } = generateTrainingData();

const trainModel = async (model: tf.Sequential, xs: tf.Tensor, ys: tf.Tensor) => {
  await model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch: number, logs?: tf.Logs) => {
        if (logs) {
          console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
        }
      },
    },
  });
};

trainModel(model, xs, ys).then(() => {
  console.log('Model trained');
});

const getBestMove = (model: tf.Sequential, matrix: Array<Array<string | null>>) => {
  const board = matrix.flat().map(cell => cell === 'R' ? 1 : cell === 'Y' ? -1 : 0);
  const prediction = model.predict(tf.tensor2d([board])) as tf.Tensor;
  const move = prediction.argMax(-1).dataSync()[0];
  return move;
};

const App: React.FC = () => {
  const [matrix, setMatrix] = useState<Array<Array<string | null>>>(initializeMatrix());
  const [currentPlayer, setCurrentPlayer] = useState<Player>('R');

  const playTurn = (col: number): void => {
    try {
      const newMatrix = addTokenToCol([...matrix.map(row => [...row])], col, currentPlayer);

      if (checkWin(newMatrix, currentPlayer)) {
        console.log(`${currentPlayer} wins!`);
        setMatrix(initializeMatrix());
      } else {
        setMatrix(newMatrix);
        setCurrentPlayer(currentPlayer === 'R' ? 'Y' : 'R');
      }
    } catch (error) {
      if (error instanceof Error) {
        console.error(error.message);
      } else {
        console.error('An unknown error occurred');
      }
    }
  };

  const playAITurn = () => {
    const col = getBestMove(model, matrix);
    playTurn(col);
  };

  useEffect(() => {
    if (currentPlayer === 'Y') {
      playAITurn();
    }
  }, [currentPlayer]);

  return (
    <div>
      <h1>Puissance 4</h1>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${numCols}, 50px)` }}>
        {Array.from({ length: numRows }, (_, rowIndex) => (
          <React.Fragment key={rowIndex}>
            {matrix.map((col, colIndex) => (
              <div
                key={colIndex + rowIndex * numCols}
                style={{
                  width: 50,
                  height: 50,
                  border: '1px solid black',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: col[rowIndex] ? (col[rowIndex] === 'R' ? 'red' : 'yellow') : 'white'
                }}
                onClick={() => playTurn(colIndex)}
              >
                {col[rowIndex]}
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default App;
