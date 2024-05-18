import React, { useEffect, useState, useCallback } from 'react';
import { createModel } from './model';
import * as tf from '@tensorflow/tfjs';
import LZString from 'lz-string';

type Player = 'R' | 'Y';

const numRows = 6;
const numCols = 7;

const initializeMatrix = (): Array<Array<string | null>> => {
  //console.log('Initializing matrix...');
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

const checkWin = (matrix: Array<Array<string | null>>, player: Player): boolean => {
  for (let col = 0; col < numCols; col++) {
    for (let row = 0; row < numRows; row++) {
      if (
        checkDirection(matrix, row, col, 0, 1, player) || // Horizontal
        checkDirection(matrix, row, col, 1, 0, player) || // Vertical
        checkDirection(matrix, row, col, 1, 1, player) || // Diagonal down-right
        checkDirection(matrix, row, col, 1, -1, player)   // Diagonal down-left
      ) {
        return true;
      }
    }
  }
  return false;
};

const isBoardFull = (matrix: Array<Array<string | null>>): boolean => {
  return matrix.every(col => col.every(cell => cell !== null));
};

const player1Model = createModel();
const player2Model = createModel();

const trainModel = async (model: tf.Sequential, xs: tf.Tensor, ys: tf.Tensor) => {
 // console.log('Training model...');
  await model.fit(xs, ys, {
    epochs: 1,
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
  console.log('Model trained.');
};

const getBestMove = (model: tf.Sequential, matrix: Array<Array<string | null>>): number => {
  const board = matrix.flat().map(cell => cell === 'R' ? 1 : cell === 'Y' ? -1 : 0);
  const prediction = model.predict(tf.tensor2d([board])) as tf.Tensor;
  const predictionData = prediction.dataSync();

  const validMoves = [];
  for (let i = 0; i < numCols; i++) {
    if (matrix[i][0] === null) {
      validMoves.push({ col: i, score: predictionData[i] });
    }
  }

  if (validMoves.length === 0) {
    throw new Error('No valid moves available');
  }

  validMoves.sort((a, b) => b.score - a.score);
  const bestMove = validMoves[0].col;

  // Ajout d'un facteur aléatoire pour la sélection des mouvements
  const randomFactor = Math.random();
  if (randomFactor > 0.7 && validMoves.length > 1) {
    return validMoves[1].col; // Sélectionner le deuxième meilleur coup avec une probabilité de 30%
  }

  return bestMove;
};

const App: React.FC = () => {
  const [matrix, setMatrix] = useState<Array<Array<string | null>>>(initializeMatrix());
  const [currentPlayer, setCurrentPlayer] = useState<Player>('R');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [gameData, setGameData] = useState<{ boards: number[][], moves: number[] }>({ boards: [], moves: [] });
  const [allGameData, setAllGameData] = useState<{ boards: number[][], moves: number[] }>({ boards: [], moves: [] });
  const [winner, setWinner] = useState<Player | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(true);
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [partiePlayed, setPartiePlayed] = useState<number>(0);

  useEffect(() => {
    const train = async () => {
      const storedGameData = localStorage.getItem('gameData');
      if (storedGameData) {
        const decompressedData = LZString.decompress(storedGameData);
        if (decompressedData) {
          try {
            const parsedData = JSON.parse(decompressedData);
            const { boards, moves } = parsedData;
            if (Array.isArray(boards) && Array.isArray(moves)) {
              const xs = tf.tensor2d(boards);
              const ys = tf.oneHot(tf.tensor1d(moves, 'int32'), numCols);
              await trainModel(player1Model, xs, ys);
              await trainModel(player2Model, xs, ys);
              setAllGameData({ boards, moves });
            }
          } catch (error) {
            console.error('Error parsing stored game data:', error);
          }
        }
      }
      //console.log('Models trained');
      setIsTraining(false);
    };
    train();
  }, []);

  const startGame = () => {
    setPartiePlayed(partiePlayed + 1);
    console.log(partiePlayed);
    setMatrix(initializeMatrix());
    setCurrentPlayer('R');
    setIsPlaying(true);
    setWinner(null);
    setGameOver(false);
    setGameData({ boards: [], moves: [] });
  };

  const playTurn = useCallback((col: number): void => {
    if (gameOver) return;

    try {
      const newMatrix = addTokenToCol([...matrix.map(row => [...row])], col, currentPlayer);
      const board = newMatrix.flat().map(cell => cell === 'R' ? 1 : cell === 'Y' ? -1 : 0);

      setGameData(prevData => ({
        boards: [...prevData.boards, board],
        moves: [...prevData.moves, col]
      }));

      if (checkWin(newMatrix, currentPlayer)) {
        setMatrix(newMatrix);
        setIsPlaying(false);
        setWinner(currentPlayer);
        setGameOver(true);

        setTimeout(() => {
          const xs = tf.tensor2d(gameData.boards);
          const ys = tf.oneHot(tf.tensor1d(gameData.moves, 'int32'), numCols);
          trainModel(currentPlayer === 'R' ? player1Model : player2Model, xs, ys);

          setAllGameData(prevData => {
            const newBoards = [...prevData.boards, ...gameData.boards];
            const newMoves = [...prevData.moves, ...gameData.moves];

            localStorage.setItem('gameData', LZString.compress(JSON.stringify({ boards: newBoards, moves: newMoves })));

            return {
              boards: newBoards,
              moves: newMoves
            };
          });

          startGame();
        }, 3000);
      } else if (isBoardFull(newMatrix)) {
        setMatrix(newMatrix);
        setIsPlaying(false);
        setWinner(null);
        setGameOver(true);

        // Si le plateau est plein, redémarrer le jeu
        setTimeout(() => {
          console.log('Board is full, restarting the game...');
          startGame();
        }, 3000);
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
  }, [currentPlayer, gameData, gameOver, matrix]);

  const playAITurn = useCallback(() => {
    if (!isPlaying) return;

    const model = currentPlayer === 'R' ? player1Model : player2Model;
    const col = getBestMove(model, matrix);
    playTurn(col);
  }, [currentPlayer, isPlaying, matrix, playTurn]);

  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(playAITurn, 500);
      return () => clearTimeout(timer);
    }
  }, [currentPlayer, isPlaying, playAITurn]);


  return (
    <div>
      <h1>Puissance 4</h1>
      {isTraining ? (
        <div>Training models, please wait...</div>
      ) : (
        <>
          <button onClick={startGame} disabled={isPlaying}>Start Game</button>
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
                  >
                    {col[rowIndex]}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
          {winner && (
            <div>
              <h2>Player {winner} wins!</h2>
              <button onClick={startGame}>Restart</button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default App;
