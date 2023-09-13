using System.Collections.Generic;
using System.Linq;

namespace AI.Base.Core
{
    public abstract class Activator: Layer
    {
        public void Activate(DataRow data)
        {
        }

        internal DataRow GetDerivativeFor(DataRow data)
        {
            var res = new double[data.Length];
            
            for (int i = 0; i < data.Length; i++)
            {
                res[i] = GetDerivativeForMember(data[i]);
            }

            return new DataRow(res);

        }

        public override void Initialize() 
        { }

        public override DataRow Process(DataRow data)
        {
            var res = new double[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                res[i] = ActivateMember(data[i]);
            }

            IntermediateValues = new List<DataRow> { new DataRow(res) };
            return IntermediateValues.FirstOrDefault();
        }

        public override List<DataRow> ProcessGroup(List<DataRow> input)
        {
            IntermediateValues = input.Select(row =>
            {
                var res = new double[row.Length];
                for (int i = 0; i < row.Length; i++)
                {
                    res[i] = ActivateMember(row[i]);
                }

                return new DataRow(res);
            }).ToList();

            return IntermediateValues;
        }

        internal abstract double ActivateMember(double member);

        internal abstract double GetDerivativeForMember(double member);

    }
}
