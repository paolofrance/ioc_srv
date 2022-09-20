#!/usr/bin/env python3
import copy
from scipy import optimize
import numpy as np
import quadprog
import rospy


def kronsum(A, B):
    if A.shape[0] is not A.shape[1]:
        print('Matrix A is not a squared nxn matrix. Kronsum is defined for matrices such that AxIm + BxIn')
    if B.shape[0] is not B.shape[1]:
        print('Matrix B is not a squared mxm matrix. Kronsum is defined for matrices such that AxIm + BxIn')

    C = np.kron(A, np.identity(B.shape[0])) + np.kron(np.identity(A.shape[0]), B)

    return C


def IOC():

    A = np.array([[0, 1],[0, -5]])
    B = np.array([[0],[0.1]])

    K_hat = np.array([65.6064, 16.5722])
    K_hat = np.reshape(K_hat, (1, 2))

    F = A - B @ K_hat

    F_kron = kronsum(F.T, F.T)

    Z1 = np.kron(np.identity(B.shape[0]), B.T) @ np.linalg.inv(F_kron)

    K_kron = np.kron(K_hat.T, K_hat.T)

    M1_ = np.concatenate((Z1, Z1 @ K_kron + np.kron(K_hat.T, np.identity((K_hat.shape[0])))),
                         axis=1)

    M1 = np.concatenate((M1_[:, 0].reshape(2, 1), M1_[:, 3].reshape(2, 1), M1_[:, 4].reshape(2, 1)), axis=1)
    print(M1)

    H = M1.T @ M1

    f = np.zeros((H.shape[1], 1)).reshape((3,))

    a_eq = np.zeros(H.shape)
    a_eq[0, 0] = 1
    b_eq = np.zeros((H.shape[0], 1)).reshape((3,))
    b_eq[0] = 1

    V = H + 0.00001 * np.eye(H.shape[0])

    theta1 = quadprog.solve_qp(V, f, a_eq, b_eq)[0]

    Q1_hat = np.diag([theta1[0], theta1[1]])
    R1_hat = np.diag([theta1[2]])

    print(Q1_hat)
    print(R1_hat)


def ioc_srv():
    rospy.init_node('ioc_srv')
    s = rospy.Service('add_two_ints', ioc, IOC)
    print("Ready to add two ints.")
    rospy.spin()


if __name__ == '__main__':

    IOC()
