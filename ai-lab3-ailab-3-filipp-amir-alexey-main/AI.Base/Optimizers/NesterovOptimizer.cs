using AI.Base.Core;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace AI.Base.Optimizers
{
    public class NesterovOptimizer : Optimizer
    {
        private readonly double learningRate;
        private readonly double momentum;
        private double velocity;

        public NesterovOptimizer(double learningRate = 0.001, double momentum = 0.9)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.velocity = 0.0;
        }

        public double Optimize(double gradient)
        {
            velocity = momentum * velocity - learningRate * gradient;
            double delta = momentum * velocity - learningRate * gradient;
            return delta;
        }

        public DataRow Optimize(DataRow gradient)
        {
            var res = new double[gradient.Length];
            for (int i = 0; i < gradient.Length; i++)
            {
                velocity = momentum * velocity - learningRate * gradient[i];
                res[i] = momentum * velocity - learningRate * gradient[i];
            }
            return new DataRow(res);
        }

        public DenseMatrix Optimize(DenseMatrix gradient)
        {
            var rows = gradient.RowCount;
            var cols = gradient.ColumnCount;
            var res = new DenseMatrix(rows, cols);

            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    velocity = momentum * velocity - learningRate * gradient[row, col];
                    res[row, col] = momentum * velocity - learningRate * gradient[row, col];
                }
            }

            return res;
        }
    }
}
