using System;

namespace AI.Base.Activators
{
    public class ReLU : Core.Activator
    {
        internal override double ActivateMember(double x)
        {
            return Math.Max(x, 0);
        }

        internal override double GetDerivativeForMember(double x)
        {
            return (x >= 0) ? 1 : 0;
        }
    }
}
