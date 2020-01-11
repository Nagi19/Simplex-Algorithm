import numpy as np
from numpy.linalg import inv

class Simplex:

    def __init__(self, A, b,c, rule: int = 0):
        self.coefMatrix = A
        self.valueMatrix = b
        self.costMatrix = c
        self.rule = rule

    def copatibilityCheck(self, rowCount, colCount):
        if colCount < rowCount:
            return False , "System Incompatibility ------ (no. of variables : {} GREATER THAN {} : no.of constraints".format(colCount, rowCount)
        if b.shape != (rowCount,):
            return False, "System Incompatibility ------ Cost Matrix_j has shape {}, expected {}.".format(b.shape, (rowCount,))
        if c.shape != (colCount,):
            return False, "System Incompatibility ------ Cost Matrix has shape {}, expected {}.".format(c.shape, (colCount,))
        return True, "Success"

    def simplexBase(self):
        rowCount, colCount = self.coefMatrix.shape[0], self.coefMatrix.shape[1]

        error, msg = self.copatibilityCheck(rowCount, colCount)
        if not error:
            return msg

        bfs1 = [0]*(colCount - rowCount) + [1  for i in range((rowCount))]
        bfs1 = np.array(bfs1)
        basic_init = set(range(colCount-rowCount,colCount))

        msg, x, basic, ofv, bfs, count = self.simplexPhaseII(bfs1, basic_init)

        if msg == 0:
            print("\n\n" +
                  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SOLUTION to the LP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" +
                    "\n\n" +
                   "Found optimal solution at x = {}. \n".format(x) +
                   "Basic variables: {}\n".format(basic) +
                   "Nonbasic variables: {}\n".format(set(range(colCount)) - basic) +
                   "Optimal value function: {}.".format(ofv))
        elif msg == 1:

            print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" +
                   "\n\n" +
                   "LP is UNBOUNDED")

        return msg, x, ofv, bfs

    def unBoundedCheck(self, rccv_q, N, B_inv):
        for item, i in zip(rccv_q, N):
            if item[0] > 0:
                a = B_inv * self.coefMatrix[:, item[1]]
                return all(i <= 0 for i in a)
        return False

    def blandsRule(self,prices, N):

        rccv =[((self.costMatrix[q] - prices * self.coefMatrix[:, q]).item(), q) for q in N]
        posRccv = [item[1] for item in rccv if item[0] > 0]
        minSub = min(posRccv) if posRccv else 0
        corRccv = 0
        for i,j in rccv:
            if minSub == j:
                corRccv = i
        rcc = [(self.costMatrix[q] - prices * self.coefMatrix[:, q]).item() for q in N]

        return all(i <= 0 for i in rcc), corRccv, minSub

    def maxCoefRule(self, prices, N):
        rccv, q = max([((self.costMatrix[q] - prices * self.coefMatrix[:, q]).item(), q) for q in N],
                     key=(lambda item: item[0]))
        rcc = [(self.costMatrix[q] - prices * self.coefMatrix[:, q]).item() for q in N]

        return all(i <=0  for i in rcc), rccv, q

    def simplexPhaseII(self,  x: np.array, basic: set, ):

        rowCount, colCount = self.coefMatrix.shape[0], self.coefMatrix.shape[1]
        B, N = list(basic), set(range(colCount)) - basic
        B_inv = inv(self.coefMatrix[:, B])

        ofv = np.dot(self.costMatrix, x)
        del basic

        count = 1
        while count < 50:
            rccv, basicwitch, nonBasicSwitch, delta, mipr = None, None, None, None, None

            prices = self.costMatrix[B] * B_inv
            rccv_q = [((self.costMatrix[q] - prices * self.coefMatrix[:, q]).item(), q) for q in N]

            print ("------------------------------------------------------------------------")
            print ("---------------------- Iteration - {} ----------------------------------".format(count))
            print ("Next Basic variables: {}\n".format(B) +
                    "Next Nonbasic variables: {}\n".format(N) +
                    "Objective value function: {}\n".format(ofv) +
                    "RCCV : {}".format(rccv_q) )

            print ("------------------------------------------------------------------------")

            count += 1

            unbounded = self.unBoundedCheck(rccv_q, N, B_inv)
            if unbounded:
                 return 1, x, set(B), None, mipr, count

            if self.rule == 0:
                """Blands AntiCycling Rule"""
                optimum, rccv, basicwitch = self.blandsRule(prices, N)

            elif self.rule == 1:
                """Maximum Coefficient Rule"""
                optimum, rccv, basicwitch = self.maxCoefRule(prices,N)

            else:
                """No Valid Rule"""
                raise ValueError ("Please input a valid Pivot Rule")

            if optimum:
                """Optimum Value Found"""
                ofv =  np.dot(prices, self.valueMatrix)
                if all(i for i in x >=0):
                   return 0, x, set(B), ofv, None, count

            """Build the next BFS"""
            bfsbuild = np.zeros(colCount)
            for i in range(rowCount):
                bfsbuild[B[i]] = (-B_inv[i, :] * self.coefMatrix[:, basicwitch]).item()

            bfsbuild[basicwitch] = 1

            mipr = [(-x[B[i]] / bfsbuild[B[i]], i) for i in range(rowCount) if bfsbuild[B[i]] < 0]

            if len(mipr) == 0:
                print("Unbounded Problem has been identified")
                return 1, x, set(B), None, bfsbuild, count

            delta, nonBasicSwitch = max(mipr, key=(lambda item: item[0]))

            """Update the variables"""
            x = np.array([var for var in (x + delta * bfsbuild)])

            """Update Objective fucntion value"""
            ofv = (ofv + delta * rccv)

            for i in set(range(rowCount)) - {nonBasicSwitch}:
                B_inv[i, :] -= bfsbuild[B[i]]/bfsbuild[B[nonBasicSwitch]] * B_inv[nonBasicSwitch, :]

            B_inv[nonBasicSwitch, :] /= -bfsbuild[B[nonBasicSwitch]]

            """Non Basic Variable update"""
            N = N - {basicwitch} | {B[nonBasicSwitch]}

            """Basic Variable update"""
            B[nonBasicSwitch] = basicwitch

        raise TimeoutError("LP is running into Infinite Loop")


if __name__ == "__main__":

    """
        Rule 0 - Bland's Anti Cyclic Rule
        Rule 1 - Maximum Coefficient Pivot rule 
    """
    rule = 0

    """
        Example 1
    """
    A = np.matrix([[0.5, -5.5,-2.5,9,1,0,0], [0.5, -1.5, -0.5,1, 0,1,0],[1,0,0,0,0,0,1]])
    b = np.array([0,0,1])
    c = np.array([10, -57,-9,-24, 0,0,0])

    """
        Example 2
    """
    A = np.matrix([[1,0,0,1,0,0], [0,1,0,0,1,0],[0,0,1,0,0,1]])
    b = np.array([2,1,6])
    c = np.array([2,3,1,0,0,0])


    """
        Example 3
    """
    # A = np.matrix([[200,80,40,1], [1,1,1,0]])
    # b = np.array([10000,50])
    # c = np.array([60,20,30,0])
    obj = Simplex(A, b, c, rule)
    obj.simplexBase()

