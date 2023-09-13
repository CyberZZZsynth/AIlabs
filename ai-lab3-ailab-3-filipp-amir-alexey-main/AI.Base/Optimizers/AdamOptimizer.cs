using AI.Base.Core;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace AI.Base.Optimizers
{
    public class AdamOptimizer: Optimizer
    {
        private readonly double learningRate;
        private readonly double beta1;
        private readonly double beta2;
        private readonly double epsilon;
        private double m;
        private double v;
        private double t;

        public AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            m = 0.0;
            v = 0.0;
            t = 1;
        }

        public double Optimize(double gradient)
        {
            m = beta1 * m + (1 - beta1) * gradient;
            v = beta2 * v + (1 - beta2) * gradient * gradient;
            double mHat = m / (1 - Math.Pow(beta1, t));
            double vHat = v / (1 - Math.Pow(beta2, t));
            double delta = -learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            t++;
            return delta;
        }

        DataRow Optimizer.Optimize(DataRow gradient)
        {
            var res = new double[gradient.Length];
            for (int i = 0; i < gradient.Length; i++)
            {
                m = beta1 * m + (1 - beta1) * gradient[i];
                v = beta2 * v + (1 - beta2) * gradient[i] * gradient[i];
                double mHat = m / (1 - Math.Pow(beta1, t));
                double vHat = v / (1 - Math.Pow(beta2, t));
                res[i] = -learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            }
            t++;
            return new DataRow(res);
        }

        DenseMatrix Optimizer.Optimize(DenseMatrix gradient)
        {

            var rows = gradient.RowCount;
            var cols = gradient.ColumnCount;
            var res = new DenseMatrix(rows, cols);

            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    m = beta1 * m + (1 - beta1) * gradient[row, col];
                    v = beta2 * v + (1 - beta2) * gradient[row, col] * gradient[row, col];
                    double mHat = m / (1 - Math.Pow(beta1, t));
                    double vHat = v / (1 - Math.Pow(beta2, t));
                    res[row, col] = -learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
                }
            }
            t++;

            return res;
        }
    }
}
