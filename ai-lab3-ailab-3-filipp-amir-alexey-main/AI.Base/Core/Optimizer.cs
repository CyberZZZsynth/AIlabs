using MathNet.Numerics.LinearAlgebra.Double;

namespace AI.Base.Core
{
    public interface Optimizer
    {
        public double Optimize(double gradient);
        public DataRow Optimize(DataRow gradient);
        public DenseMatrix Optimize(DenseMatrix gradient);
    }
}
