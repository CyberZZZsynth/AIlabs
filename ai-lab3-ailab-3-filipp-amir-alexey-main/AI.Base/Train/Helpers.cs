using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AI.Base.Train
{
    public static class Helpers
    {
        public static IEnumerable<R> Map<T,R>(IEnumerable<T> elems, Func<T,R> view)
        {
            var enumerator = elems.GetEnumerator();
            while(enumerator.MoveNext())
            {
                yield return view(enumerator.Current);
            }
        }

        public static IEnumerable<T[]> Batching<T>(IEnumerable<T> elems, int size)
        {
            var enumerator = elems.GetEnumerator();
            while (true)
            {
                var current = new LinkedList<T>();
                while(current.Count < size && enumerator.MoveNext())
                {
                    current.AddLast(enumerator.Current);
                }
                if(current.Count == size)
                {
                    yield return current.ToArray();
                }
                else
                {
                    if(current.Count > 0)
                    {
                        yield return current.ToArray();
                    }
                    yield break;
                }
            }
        }

        public static IEnumerable<T> Shuffle<T>(IEnumerable<T> elems)
        {
            var random = new Random();
            var all = elems.ToArray();
            for(int i = 0; i < all.Length; i++)
            {
                var i1 = random.Next(all.Length);
                var i2 = random.Next(all.Length);
                var t = all[i1];
                all[i1] = all[i2];
                all[i2] = t;
            }
            return all;
        }
    }
}
