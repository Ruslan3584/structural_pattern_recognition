import numpy as np

sudoku_dataset = [
[
# solved by or-and
np.array([
 [2, 6, 4, 0, 0, 3, 0, 0, 1],
 [0, 9, 5, 8, 0, 0, 3, 6, 0],
 [1, 0, 0, 0, 5, 0, 7, 0, 0],
 [0, 0, 0, 0, 2, 5, 6, 9, 0],
 [0, 0, 0, 4, 0, 8, 0, 0, 0],
 [0, 8, 2, 3, 7, 0, 0, 0, 0],
 [0, 0, 9, 0, 3, 0, 0, 0, 4],
 [0, 1, 6, 0, 0, 2, 9, 7, 0],
 [3, 0, 0, 9, 0, 0, 5, 2, 6]
]),
np.array([
[2, 6, 4, 7, 9, 3, 8, 5, 1],
[7, 9, 5, 8, 1, 4, 3, 6, 2],
[1, 3, 8, 2, 5, 6, 7, 4, 9],
[4, 7, 3, 1, 2, 5, 6, 9, 8],
[9, 5, 1, 4, 6, 8, 2, 3, 7],
[6, 8, 2, 3, 7, 9, 4, 1, 5],
[5, 2, 9, 6, 3, 7, 1, 8, 4],
[8, 1, 6, 5, 4, 2, 9, 7, 3],
[3, 4, 7, 9, 8, 1, 5, 2, 6]
])

],
[
# solved by or-and
np.array([
[ 6, 2, 0, 9, 5, 1, 0, 0, 0],     
[ 0, 0, 5, 4, 0, 0, 1, 9, 6],       
[ 9, 0, 1, 0, 7, 0, 0, 2, 8],    
[ 7, 0, 4, 2, 0, 0, 8, 0, 1],    
[ 0, 8, 0, 6, 1, 0, 4, 3, 0],   
[ 1, 0, 9, 0, 4, 3, 2, 0, 0],     
[ 8, 1, 0, 7, 0, 4, 0, 0, 2],    
[ 0, 5, 7, 0, 0, 9, 3, 8, 0],   
[ 0, 9, 0, 0, 8, 2, 6, 1, 0],    
]),
np.array([
[6, 2, 8, 9, 5, 1, 7, 4, 3],
[3, 7, 5, 4, 2, 8, 1, 9, 6],
[9, 4, 1, 3, 7, 6, 5, 2, 8],
[7, 3, 4, 2, 9, 5, 8, 6, 1],
[5, 8, 2, 6, 1, 7, 4, 3, 9],
[1, 6, 9, 8, 4, 3, 2, 7, 5],
[8, 1, 6, 7, 3, 4, 9, 5, 2],
[2, 5, 7, 1, 6, 9, 3, 8, 4],
[4, 9, 3, 5, 8, 2, 6, 1, 7]
])
],

    
[
# can't find solution
np.array([
 [0, 0, 5, 2, 0, 8, 1, 0, 0],      
 [0, 4, 2, 0, 6, 0, 0, 8, 0],       
 [9, 0, 7, 0, 0, 0, 2, 3, 6],     
 [6, 0, 0, 8, 0, 5, 0, 0, 9],       
 [0, 5, 0, 0, 9, 0, 0, 4, 0],      
 [8, 0, 0, 4, 0, 7, 0, 0, 5],      
 [4, 2, 1, 0, 0, 0, 9, 0, 3],       
 [0, 3, 0, 0, 5, 0, 4, 6, 0],      
 [0, 0, 6, 1, 0, 3, 8, 0, 0]      
]),
np.array([
 [3, 6, 5, 2, 7, 8, 1, 9, 4],   
 [1, 4, 2, 3, 6, 9, 5, 8, 7],    
 [9, 8, 7, 5, 1, 4, 2, 3, 6],     
 [6, 7, 4, 8, 2, 5, 3, 1, 9],       
 [2, 5, 3, 6, 9, 1, 7, 4, 8],       
 [8, 1, 9, 4, 3, 7, 6, 2, 5],       
 [4, 2, 1, 7, 8, 6, 9, 5, 3],       
 [7, 3, 8, 9, 5, 2, 4, 6, 1],      
 [5, 9, 6, 1, 4, 3, 8, 7, 2]    
])
],

[
# can't find solution
np.array([
 [1, 0, 9, 0, 0, 0, 2, 0, 0],  
 [0, 6, 7, 0, 9, 8, 3, 0, 0],   
 [0, 4, 0, 2, 0, 5, 0, 6, 0],    
 [7, 2, 0, 8, 0, 0, 0, 1, 0],     
 [0, 0, 0, 6, 1, 7, 0, 0, 0],    
 [0, 3, 0, 0, 0, 4, 0, 5, 8],    
 [0, 9, 0, 7, 0, 1, 0, 2, 0],     
 [0, 0, 2, 4, 8, 0, 5, 7, 0],    
 [0, 0, 8, 0, 0, 0, 1, 0, 3]   
]),
np.array([
 [1, 5, 9, 3, 4, 6, 2, 8, 7], 
 [2, 6, 7, 1, 9, 8, 3, 4, 5],    
 [8, 4, 3, 2, 7, 5, 9, 6, 1],      
 [7, 2, 4, 8, 5, 3, 6, 1, 9],        
 [9, 8, 5, 6, 1, 7, 4, 3, 2],       
 [6, 3, 1, 9, 2, 4, 7, 5, 8],        
 [5, 9, 6, 7, 3, 1, 8, 2, 4],      
 [3, 1, 2, 4, 8, 9, 5, 7, 6],        
 [4, 7, 8, 5, 6, 2, 1, 9, 3]
])
],

[
np.array([
[0, 0, 6, 0, 0, 1, 0, 0, 9],
[2, 0, 0, 0, 8, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 2, 5, 0],
[6, 0, 0, 1, 5, 0, 0, 0, 8],
[3, 0, 0, 6, 0, 4, 0, 0, 2],
[1, 0, 0, 0, 2, 7, 0, 0, 6],
[0, 3, 2, 0, 0, 0, 4, 0, 0],
[0, 0, 0, 0, 7, 0, 0, 0, 1],
[5, 0, 0, 3, 0, 0, 8, 0, 0]
]),
np.array([
[4, 5, 6, 2, 3, 1, 7, 8, 9],
[2, 7, 3, 5, 8, 9, 6, 1, 4],
[8, 9, 1, 7, 4, 6, 2, 5, 3],
[6, 2, 7, 1, 5, 3, 9, 4, 8],
[3, 8, 5, 6, 9, 4, 1, 7, 2],
[1, 4, 9, 8, 2, 7, 5, 3, 6],
[7, 3, 2, 9, 1, 8, 4, 6, 5],
[9, 6, 8, 4, 7, 5, 3, 2, 1],
[5, 1, 4, 3, 6, 2, 8, 9, 7]
])


]

]