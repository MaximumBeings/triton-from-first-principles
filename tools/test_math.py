"""Independent CPU binary64 arithmetic for worked illustrations; not kernel tests."""
import math
import unittest


def online(scores, values, block):
    maximum, denominator, numerator = -math.inf, 0.0, 0.0
    for start in range(0, len(scores), block):
        ss, vv = scores[start:start+block], values[start:start+block]
        new = max(maximum, max(ss))
        scale = math.exp(maximum-new)
        weights = [math.exp(s-new) for s in ss]
        denominator = scale*denominator+sum(weights)
        numerator = scale*numerator+sum(p*v for p,v in zip(weights,vv))
        maximum = new
    return numerator/denominator


class WorkedMath(unittest.TestCase):
    def test_sector_requests_versus_unique_footprint(self):
        fields=[3,4,5,6]
        requests=[{(i*28+field*4)//32 for i in range(32)} for field in fields]
        self.assertEqual([len(x) for x in requests],[28]*4)
        self.assertEqual(sum(map(len,requests)),112)
        self.assertEqual(len(set.union(*requests)),28)
        self.assertEqual(32*4*4/(28*32),4/7)

    def test_discount_error_contract_attribution(self):
        inputs=zip([100,100,100,80,120,100],[100,90,110,100,100,100],[1,.5,2,1,1,.01],[.05,.05,.03,.05,.05,.05],[.2,.3,.25,.4,.15,.2])
        relative=[]
        cdf=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
        for S,K,T,r,vol in inputs:
            d1=(math.log(S/K)+(r+vol*vol/2)*T)/(vol*math.sqrt(T));d2=d1-vol*math.sqrt(T)
            correct=S*cdf(d1)-K*math.exp(-r*T)*cdf(d2)
            wrong=S*cdf(d1)-K*cdf(d2)
            relative.append(100*(wrong-correct)/correct)
        self.assertEqual(max(range(6),key=lambda i:abs(relative[i])),0)
        self.assertAlmostEqual(relative[0],-26.116,delta=.01)
        self.assertAlmostEqual(relative[2],-19.947,delta=.01)

    def test_stride_units(self):
        offsets = [[i*5+j for j in range(3)] for i in range(2)]
        self.assertEqual(offsets, [[0,1,2],[5,6,7]])
        self.assertEqual(offsets[1][2]*4, 28)

    def test_mask_coverage_and_causality(self):
        offsets=[p*8+lane for p in range(2) for lane in range(8) if p*8+lane<13]
        self.assertEqual(offsets,list(range(13)))
        self.assertEqual(sum(j<=i for i in range(4) for j in range(4)),10)

    def test_gemm_full_reduction(self):
        a=[[1,2],[3,4]];b=[[5,6],[7,8]]
        c=[[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)]for i in range(2)]
        self.assertEqual(c,[[19,22],[43,50]])

    def test_online_rescaling(self):
        for scores,values in [([math.log(2),0,math.log(2)],[1,3,5]),([-20,5,1,9],[2,-3,4,1])]:
            m=max(scores);p=[math.exp(s-m)for s in scores]
            reference=sum(x*v for x,v in zip(p,values))/sum(p)
            for block in (1,2,3,4):self.assertAlmostEqual(online(scores,values,block),reference,places=13)
        self.assertAlmostEqual(online([math.log(2),0,math.log(2)],[1,3,5],2),3)

    def test_gelu_chain_rule(self):
        phi=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
        gelu=lambda x:x*phi(x)
        for x in (-3,-0.5,0,1,3):
            analytic=phi(x)+x*math.exp(-x*x/2)/math.sqrt(2*math.pi)
            h=1e-5;numeric=(gelu(x+h)-gelu(x-h))/(2*h)
            self.assertAlmostEqual(analytic,numeric,places=8)
            self.assertAlmostEqual(-2*analytic,( -2*gelu(x+h)+2*gelu(x-h))/(2*h),places=8)

    def test_population_and_sample_variance(self):
        x=[1,2,3,4];mean=sum(x)/len(x);ss=sum((v-mean)**2 for v in x)
        self.assertEqual(mean,2.5);self.assertEqual(ss/len(x),1.25)
        self.assertAlmostEqual(ss/(len(x)-1),5/3)
        self.assertEqual(sum([3,-1,2,4,0,0,0,0]),8)

    def test_option_formula_and_boundary(self):
        S=K=100;T=1;r=.05;vol=.2
        cdf=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
        d1=(math.log(S/K)+(r+vol*vol/2)*T)/(vol*math.sqrt(T));d2=d1-vol*math.sqrt(T)
        value=S*cdf(d1)-K*math.exp(-r*T)*cdf(d2)
        self.assertAlmostEqual(value,10.450583572185565,places=10)
        self.assertEqual(max(S-K,0),0)
        self.assertAlmostEqual(max(S-K*math.exp(-r*T),0),4.877057549928594,places=10)

    def test_monte_carlo_arithmetic_only(self):
        payoffs=[max(s-100,0)for s in [80,100,120]]
        self.assertEqual(payoffs,[0,0,20]);self.assertEqual(sum(payoffs)/3,20/3)

if __name__=='__main__':unittest.main()
