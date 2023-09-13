using AI.Base.Core;
using MathNet.Numerics.LinearAlgebra.Double;

namespace AI.Base.Optimizers
{
    public class SGD : Optimizer
    {
        private double learningRate;

        public SGD(double learningRate = 0.01)
        {
            this.learningRate = learningRate;
        }


        public double Optimize(double gradient)
        {
            return learningRate * gradient;
        }

        public DataRow Optimize(DataRow gradient)
        {
            var res = new double[gradient.Length];
            for (int i = 0; i < gradient.Length; i++)
            {
                res[i] = learningRate * gradient[i];
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
                    res[row, col] = learningRate * gradient[row, col];
                }
            }

            return res;
        }
    }
}
