import numpy as np

a = np.array([[0, 0, 0, 0],[0, 1, 1, 0],[1, 0, 1, 0],[0, 0, 0, 0]])
b = np.where(a==True)
# print(b)

def getValidArea(ndarray):
    row, col = np.where(ndarray == True)
    minRow, maxRow = row.min(), row.max()
    minCol, maxCol = col.min(), col.max()
    return [minRow, maxRow, minCol, maxCol]