using AI.Base.Core;

namespace AI.Base.LossFunctions
{
    public class MSE: LossFunction
    {
        public double GetFunctionFor(DataRow predictedValue, DataRow expectedValue)
        {
            var sum = 0.0;
            for (int i = 0; i < predictedValue.Length; i++)
            {
                sum += (predictedValue[i] - expectedValue[i]) * (predictedValue[i] - expectedValue[i]);
            }

            return sum / predictedValue.Length;
        }

        public DataRow GetDerivativeFor(DataRow predictedValue, DataRow expectedValue)
        {
            var res = new double[predictedValue.Length];
            for (int i = 0; i < predictedValue.Length; i++)
            {
                res[i] = (predictedValue[i] - expectedValue[i]) * 2 / predictedValue.Length;
            }

            return new DataRow(res);
        }
    }
}
