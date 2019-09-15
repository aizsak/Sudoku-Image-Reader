

"""
sudoku.py contains the class sudoku. Each instance represent sudoku puzzles and can be initialized from a list, dict
or string. Contains methods to solve sudokus, copy sudokus, print sudokus. Also includes a couple methods to 
independently check the accuracy of the sudoku solver.

"""
class Sudoku:
    
    sudRange = [(i,j) for j in range(9) for i in range(9)]
    
    def __init__(self, puz = [[0]*9]*9):
    #Note, sudoku puzzle represented by a dict where values are sets containing all possible digits for that cell. 
        if isinstance(puz, list):
            self.puzzle = {(i, j): self.digitToSet(puz[i][j]) for (i, j) in self.sudRange}
            
        elif isinstance(puz, str):
            self.readSudoku(puz)
            
        elif isinstance(puz, dict):
            self.fromDict(puz)

        
        
    def digitToSet(self, cell):
        #Parameters: cell, an int
        #Output: a set containing all possible values a sudoku puzzle cell might have if it contains cell
        if cell in range(1,10):
            return {cell}
        else:
            return set(range(1,10))
        
        
    def strToSet(self, cell):
        #Parameters: cell, a string
        #Output: a set containing all possible values a sudoku puzzle cell might have if it contains cell  
        
        if cell in '123456789':
            return {int(cell)}
        else:
            return set(range(1,10))
        
        
    def indToStr(self, i, j):
        #Parameters: i, j, ints, the index of a cell
        #Output: a string, the value of that cell
        
        cell = self.puzzle[(i,j)]
        
        if len(cell) == 1 and list(cell)[0] in range(1,10):
            return str(list(cell)[0])
        else:
            return '0'
    
    
    def readSudoku(self, strSud):
        #Parameters: strSud, a 81 digit string representing a sudoku  puzzle. 
        
        self.puzzle = {(i,j): self.strToSet(strSud[j + 9 * i]) for (i, j) in self.sudRange}

         
    
    def __str__(self):      
        lines = ''
        for i in range(9):
            for j in range(9):
                lines = lines + self.indToStr(i, j) + ' '
                if j in [2,5]:
                    lines = lines + ('| ')  
            lines = lines + '\n'
            if i in [2,5]:
                lines = lines + ('-'*22)  + '\n'
        return lines
    
        
    def fromDict(self, dictSud):
        #Parameters: dictSud, a dict. Changes puzzle according to dictSud.
        self.puzzle = {position: dictSud[position] for position in self.sudRange}
        
    def copy(self): 
        #Returns a copy of this Sudoku instance
        copied = Sudoku()
        copied.fromDict(self.puzzle)
        return copied
   

    def solver(self):
        #Outputs: a dict representing the solution to this sudoku puzzle.
        puzzle = self.puzzle
        cells =  [[i,j] for i in range(9) for j in range(9)]

        neighbors = {cell:{(i,j) for (i,j) in self.sudRange if cell[0] == i or cell[1] == j or 
                           (cell[0]//3 == i//3 and cell[1]//3 ==j//3)} - {cell} for cell in self.sudRange}
        
        #checkNeighbors: a list of all positions in the sudoku puzzle that may contribute new constraints to their neighbors
        checkNeighbors = [i for i in puzzle if len(puzzle[i]) == 1]
        
        def propagateConstraints(puzzle, checkNeighbors):
            while len(checkNeighbors) > 0:
                position = checkNeighbors.pop()
                for nbor in neighbors[position]:
                    numBefore = len(puzzle[nbor])
                    puzzle[nbor] = puzzle[nbor] - puzzle[position]
                    numAfter = len(puzzle[nbor])
                    if numAfter == 0:
                        return False
                    if numBefore == 2 and numAfter == 1:
                        checkNeighbors = checkNeighbors + [nbor]
            return puzzle

        puzzle = propagateConstraints(puzzle, checkNeighbors)
        if puzzle is False:
            return False

        def recursiveSolver(puzzle):
        # First we find a cell with minimal possible values, guess a value, then propogate constraints. Recurse to see if this leads
        #to a solved puzzle. If not try the other possible values in that cell.
        
            unsolved = [i for i in puzzle if len(puzzle[i]) > 1]
            if len(unsolved) == 0:
                return puzzle
            minIndex = min(unsolved, key= lambda i: len(puzzle[i]))

            for testValue in puzzle[minIndex]:
                puzzle[minIndex] = {testValue}
                puz = propagateConstraints(puzzle.copy(),[minIndex])
                if puz is not False:
                    unsolved = [i for i in puz if len(puz) > 1]
                    if len(unsolved) == 0:
                        return puz
                    else:
                        puz = recursiveSolver(puz.copy())
                        if puz is not False:
                            return puz
            return False

        return recursiveSolver(puzzle.copy())    

        
    #Next two methods only used as a secondary  test of the solver method.
    
    def comesFrom(self, original):
        #Checks that this Sudoku puzzle is derived from an original Sudoku by only filling in blank cells.
    
        origin = original.puzzle
        startChecks = [self.puzzle[i] == origin[i] for i in origin if len(origin[i]) > 1]
        return min(startChecks)

    def checkSudoku(self): 
    #Check if a sudoku puzzle satisfies the constraints of a sudoku puzzle
        rows = [[(i,j) for j in range(9)] for i in range (9)]
        cols = [[(i,j) for i in range(9)] for j in range (9)]
        boxes =  [[(3*i+k,3*j+l) for k in range(3) for l in range(3)] for i in range(3) for j in range(3)]
        rcb = rows + cols + boxes
        neighbors = {cell: {} for cell in self.sudRange}
        for cell in self.sudRange:
            for group in rcb:
                if cell in group:
                    neighborhood = set(group) - {cell}
                    neighbors[cell] = neighborhood.union(neighbors[cell]) 
                    
        for cell in self.sudRange:
            for neighbor in neighbors[cell]:
                if self.puzzle[cell] == self.puzzle[neighbor]:
                    print(cell, neighbor)
                    return False
        
        #Check every neighborhood of self.puzzle contains all the digits from 1 to 9.
        for group in rcb:
            seen = set()
            for cell in group:
                seen = seen | self.puzzle[cell]
            if seen != set(range(1,10)):
                print(seen)
                return False
        return True

