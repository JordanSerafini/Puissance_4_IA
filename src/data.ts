import * as tf from '@tensorflow/tfjs';

type Player = 'R' | 'Y';

const numRows = 6;
const numCols = 7;

const initializeMatrix = (): Array<Array<string | null>> => {
  return Array(numCols).fill(null).map(() => Array(numRows).fill(null));
};

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

const checkWin = (matrix: Array<Array<string | null>>, player: Player): boolean => {
  const checkDirection = (startRow: number, startCol: number, deltaRow: number, deltaCol: number): boolean => {
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

  for (let col = 0; col < numCols; col++) {
    for (let row = 0; row < numRows; row++) {
      if (
        checkDirection(row, col, 0, 1) || 
        checkDirection(row, col, 1, 0) ||
        checkDirection(row, col, 1, 1) || 
        checkDirection(row, col, 1, -1) 
      ) {
        return true;
      }
    }
  }
  return false;
};

export const generateTrainingData = (player1Model: tf.Sequential, player2Model: tf.Sequential) => {
  const inputs: number[][] = [];
  const outputs: number[][] = [];

  const simulateGame = () => {
    const matrix = initializeMatrix();
    let currentPlayer: Player = 'R';
    let moveCount = 0;

    while (moveCount < numCols * numRows) {
      const availableCols = matrix.map((col, index) => col[0] === null ? index : -1).filter(index => index !== -1);
      const board = matrix.flat().map(cell => cell === 'R' ? 1 : cell === 'Y' ? -1 : 0);
      const model = currentPlayer === 'R' ? player1Model : player2Model;
      const prediction = model.predict(tf.tensor2d([board])) as tf.Tensor;
      let move = prediction.argMax(-1).dataSync()[0];

      if (matrix[move][0] !== null) {
        move = availableCols[Math.floor(Math.random() * availableCols.length)];
      }

      addTokenToCol(matrix, move, currentPlayer);

      inputs.push(board);
      const output = Array(numCols).fill(0);
      output[move] = 1;
      outputs.push(output);

      if (checkWin(matrix, currentPlayer)) {
        break;
      }

      currentPlayer = currentPlayer === 'R' ? 'Y' : 'R';
      moveCount++;
    }
  };

  for (let i = 0; i < 1000; i++) {
    simulateGame();
  }

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(outputs);

  return { xs, ys };
};
