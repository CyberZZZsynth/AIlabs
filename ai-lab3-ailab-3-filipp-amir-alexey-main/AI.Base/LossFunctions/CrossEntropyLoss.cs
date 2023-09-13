using AI.Base.Core;
using System.Collections.Generic;

namespace AI.Base.LossFunctions
{
    public class CrossEntropyLoss: LossFunction
    {
        public double GetFunctionFor(DataRow predictedValue, DataRow expectedValue)
        {
            var z = SoftMax(predictedValue);
            var res = 0.0;

            for (int i = 0; i < z.Length; i++)
            {
                res -= expectedValue[i] * System.Math.Log(z[i]);
            }

            return res;
        }

        public DataRow GetDerivativeFor(DataRow predictedValue, DataRow expectedValue)
        {
            return SoftMax(predictedValue) - expectedValue;
        }
        
        public DataRow SoftMax(DataRow data)
        {
            var res = new double[data.Length];
            var exps = new double[data.Length];
            var sum = 0.0;

            for (var i = 0; i < data.Length; i++)
            {
                exps[i] = System.Math.Exp(data[i]);
                sum += exps[i];
            }

            for (var i = 0; i < data.Length; i++)
            {
                res[i] = exps[i] / sum;
            }

            return new DataRow(res);
        }

        public List<DataRow> SoftMax(List<DataRow> data)
        {
            var reslist = new List<DataRow>();

            foreach (DataRow dataRow in data)
            {
                var res = new double[dataRow.Length];
                var exps = new double[dataRow.Length];
                var sum = 0.0;

                for (var i = 0; i < dataRow.Length; i++)
                {
                    exps[i] = System.Math.Exp(dataRow[i]);
                    sum += exps[i];
                }

                for (var i = 0; i < dataRow.Length; i++)
                {
                    res[i] = exps[i] / sum;
                }

                reslist.Add(new DataRow(res));
            }
            

            return reslist;
        }
    }
}
