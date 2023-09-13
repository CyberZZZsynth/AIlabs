using AI.Base.Core;
using System.Collections.Generic;

namespace AI.Base
{
    public abstract class Layer
    {
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        public List<DataRow> IntermediateValues { get; set; }

        public abstract void Initialize();
        public abstract DataRow Process(DataRow input);
        public abstract List<DataRow> ProcessGroup(List<DataRow> input);
    }
}
