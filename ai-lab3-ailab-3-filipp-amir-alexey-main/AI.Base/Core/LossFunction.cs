namespace AI.Base.Core
{
    public interface LossFunction
    {
        public double GetFunctionFor(DataRow predictedValue, DataRow expectedValue);
        public DataRow GetDerivativeFor(DataRow predictedValue, DataRow expectedValue);
    }
}
