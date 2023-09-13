using System;

namespace AI.Base.Activators
{
    public class HyperTan : Core.Activator
    {
        internal override double ActivateMember(double x)
        {
            return 2.0 / (1.0 + Math.Exp(-x * 2)) - 1.0;
        }

        internal override double GetDerivativeForMember(double x)
        {
            return 1 - Math.Pow(ActivateMember(x), 2);
        }
    }
}
