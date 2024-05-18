import React from 'react';

// Types pour les joueurs
type Player = '1' | '2';

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

const App: React.FC = () => {
  const [matrix, setMatrix] = React.useState<Array<Array<string | null>>>(initializeMatrix());
  const [currentPlayer, setCurrentPlayer] = React.useState<Player>('R');

  const playTurn = (col: number): void => {
    try {
      const newMatrix = addTokenToCol([...matrix.map(row => [...row])], col, currentPlayer);

      if (checkWin(newMatrix, currentPlayer)) {
        console.log(`${currentPlayer} wins!`);
        // Reset game or handle win
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
