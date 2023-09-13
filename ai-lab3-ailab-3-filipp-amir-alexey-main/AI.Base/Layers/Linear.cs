using AI.Base.Core;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using System.Linq;

namespace AI.Base.Layers
{
    public class Linear : Layer
    {
        public DenseMatrix Weights { get; set; }
        public DataRow Bias { get; set; }


        public override void Initialize()
        {
            Bias = DataRow.CreateRandom(OutputSize, new MathNet.Numerics.Distributions.ContinuousUniform(-1, 1));
            Weights = DenseMatrix.CreateRandom(OutputSize, InputSize, new MathNet.Numerics.Distributions.ContinuousUniform(-1, 1));
        }

        public override DataRow Process(DataRow input)
        {
            IntermediateValues = new List<DataRow>() { new DataRow((Weights * input.Data + Bias.Data).AsArray()) };
            return IntermediateValues.FirstOrDefault();
        }

        public override List<DataRow> ProcessGroup(List<DataRow> input)
        {
            IntermediateValues = input.Select(row => new DataRow((Weights * row.Data + Bias.Data).AsArray())).ToList();
            return IntermediateValues;
        }
    }
}
