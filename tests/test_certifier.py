"""Checks of global-error bounds, supported gates and conservative refusals."""
import math
import numpy as np
import pytest
from mpmath import mp
from deqcert import (EncoderSpec, EXAMPLES, certify, evaluate_surrogate,
                     evaluate_enclosure, extract_surrogate, simulate)
from deqcert.validated import lightcone


def spec(gates, n=1, d=1, readouts=None):
    return EncoderSpec('test', n, d, gates, readouts or ['Z0'])


def test_base_against_closed_form_and_validated_enclosures():
    s=EXAMPLES['hqz_base'](); p={k: .37 for k in s.param_names()}; r=certify(s,p)
    assert r.verdict.startswith('CERTIFIED') and r.certificate_error < 1e-12
    x=np.random.RandomState(4).uniform(-20,20,(25,4))
    with mp.workdps(80):
        a=b=mp.mpf(.37)
        for sr, offset in zip(r.basis,[0,2]):
            for row,(lo,hi) in zip(x,evaluate_enclosure(sr,x)):
                u,v=map(mp.mpf,row[offset:offset+2])
                expected=(mp.cos(b/2)**2*mp.cos(u)+mp.sin(b/2)**2*mp.cos(u)*mp.cos(v)
                          -.5*mp.sin(b)*mp.cos(a)*mp.sin(u)*(1-mp.cos(v)))
                assert mp.mpf(lo) <= expected <= mp.mpf(hi)


def test_dependent_channels_need_no_rank_assumption():
    s=spec([('rot','Y',0,('data',0,1.)),('rot','Y',0,('data',0,2.))]); r=certify(s,{})
    assert r.verdict.startswith('CERTIFIED')
    x=np.linspace(-100,100,21)[:,None]
    assert np.max(np.abs(evaluate_surrogate(r.basis[0],x)-np.cos(3*x[:,0]))) < 1e-12
    assert r.cost['construction_queries']==9


def test_weak_input_dependence_is_retained_over_real_domain():
    s=spec([('rot','Y',0,('data',0,1e-8))]); r=certify(s,{})
    assert r.verdict.startswith('CERTIFIED') and len(r.basis[0]['channels'])==1
    x=np.array([[np.pi/1e-8]]); lo,hi=evaluate_enclosure(r.basis[0],x)[0]
    with mp.workdps(80):
        expected=mp.cos(mp.mpf(1e-8)*mp.mpf(x[0,0]))
        assert mp.mpf(lo) <= expected <= mp.mpf(hi)
    assert hi < -.999999


def test_pruning_error_included_and_tolerance_enforced():
    s=spec([('rot','Y',0,('data',0,1.)),('rot','Y',0,('fixed',float(np.arccos(5e-9))))])
    r=certify(s,{},prune=1e-8,tolerance=1e-12)
    deleted=certify(s,{},prune=1.,tolerance=1e-12)
    assert r.verdict.startswith('CERTIFIED')
    assert deleted.verdict.startswith('SURROGATE FOUND') and deleted.certificate_error > .9
    assert deleted.basis[0]['n_terms']==0
    x=np.array([[0.]])
    assert abs(evaluate_surrogate(r.basis[0],x)[0]-simulate(s,x,{})['Z0'][0]) < 1e-14


def test_single_weak_fourier_mode_pruning():
    s=spec([('rot','Y',0,('fixed',math.pi/2-1e-4)),('rot','X',0,('data',0,1.)),
            ('rot','Y',0,('fixed',-math.pi/2+1e-4))])
    r=certify(s,{},prune=1e-8,tolerance=1e-12)
    assert r.verdict.startswith('SURROGATE FOUND') and r.certificate_error > 1e-9


def test_controlled_rotation_requires_half_frequencies():
    s=spec([('h',0),('crot','X',0,1,('data',0,1.)),('h',0)],n=2); r=certify(s,{})
    assert r.basis[0]['half_angle']==[True]
    x=np.linspace(-20,20,17)[:,None]
    assert np.max(np.abs(evaluate_surrogate(r.basis[0],x)-np.cos(x[:,0]/2))) < 1e-12


def test_frozen_trainable_scale_is_certifiable():
    s=spec([('rot','X',0,('data',0,('param','scale')))]); r=certify(s,{'scale':.713})
    assert r.verdict.startswith('CERTIFIED') and r.freq_class=='unbounded'
    x=np.linspace(-20,20,17)[:,None]
    assert np.max(np.abs(evaluate_surrogate(r.basis[0],x)-np.cos(.713*x[:,0]))) < 1e-12


def test_budget_refusal_precedes_every_circuit_evaluation(monkeypatch):
    import deqcert.validated as v
    import deqcert.certifier as c
    def forbidden(*args,**kwargs):
        raise AssertionError('evaluation before budget refusal')
    monkeypatch.setattr(v,'_sample',forbidden)
    monkeypatch.setattr(c,'effective_dimension',forbidden)
    monkeypatch.setattr(c,'dla_dimension',forbidden)
    r=certify(EXAMPLES['vqc_facedet']())
    assert r.verdict.startswith('INCONCLUSIVE') and r.cost['construction_queries']==0 and not r.basis


def test_manual_K_never_claims_a_certificate():
    s=spec([('rot','X',0,('data',0,1.))])
    assert certify(s,{},K=2).verdict.startswith('INCONCLUSIVE')
    fit=extract_surrogate(s,'Z0',{},K=2)
    assert fit['certified'] is False


def test_constant_readout_with_no_channels():
    r=certify(spec([('rot','Y',0,('fixed',.37))],d=0),{})
    assert r.verdict.startswith('CERTIFIED')
    assert abs(evaluate_surrogate(r.basis[0],np.empty((1,0)))[0]-np.cos(.37))<1e-14


@pytest.mark.parametrize('gate', [('unknown',0),('rot','X',0,('data',0,float('nan'))),
                                  ('rot','X',0,('fixed',2**53+1))])
def test_invalid_inputs_rejected(gate):
    with pytest.raises(ValueError):
        certify(spec([gate]),{})


def test_missing_deployed_parameters_rejected():
    with pytest.raises(ValueError):
        certify(spec([('rot','X',0,('param','theta'))]),{})


def test_structural_lightcone_matches_full_simulation():
    s=EXAMPLES['hqz_cross'](); p={k:.71 for k in s.param_names()}
    x=np.random.RandomState(1).uniform(0,6,(10,4))
    for readout in s.readouts:
        cone=lightcone(s,readout)
        assert np.max(np.abs(simulate(s,x,p)[readout]-simulate(cone,x,p)[cone.readouts[0]]))<1e-14


def test_supported_gates_against_independent_qiskit_statevector():
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Pauli
    gates=[('h',0),('rot','X',1,('data',0,1.)),('crot','Y',0,1,('data',1,.73)),
           ('crot','Z',1,0,('fixed',.29)),('cnot',0,1),
           ('rot','Y',0,('fixed',.41)),('rot','Z',1,('fixed',-.22))]
    s=spec(gates,n=2,d=2,readouts=['Z0','Z0Z1']); r=certify(s,{})
    assert r.verdict.startswith('CERTIFIED')
    for row in np.random.RandomState(7).uniform(-4,4,(8,2)):
        qc=QuantumCircuit(2)
        qc.h(0);qc.rx(row[0],1);qc.cry(.73*row[1],0,1);qc.crz(.29,1,0)
        qc.cx(0,1);qc.ry(.41,0);qc.rz(-.22,1)
        state=Statevector.from_instruction(qc)
        for sr,pauli in zip(r.basis,['IZ','ZZ']):
            assert abs(evaluate_surrogate(sr,row)[0]-state.expectation_value(Pauli(pauli)).real)<2e-13


def test_importer_symbolic_parsing_and_refusal():
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from deqcert import from_qiskit
    x,t=Parameter('x'),Parameter('t')
    qc=QuantumCircuit(1);qc.rx(2*t+.3,0)
    with pytest.raises(NotImplementedError):
        from_qiskit(qc,[],['Z0'])
    qc=QuantumCircuit(1);qc.rx(x+x*(x-.37)*(x-1),0)
    assert from_qiskit(qc,[x],['Z0']).gates[0][-1][0]=='data_poly'
    qc=QuantumCircuit(1);qc.rx(t*x+.2,0)
    with pytest.raises(NotImplementedError):
        from_qiskit(qc,[x],['Z0'])
    qc=QuantumCircuit(1,1);qc.measure(0,0);qc.h(0)
    with pytest.raises(NotImplementedError):
        from_qiskit(qc,[],['Z0'])


@pytest.mark.slow
def test_ZZ_validated_model_against_numpy_statevector():
    s=EXAMPLES['zz_feature_map'](); r=certify(s)
    assert r.verdict.startswith('CERTIFIED') and r.certificate_error < 1e-10
    x=np.random.RandomState(3).uniform(-2,8,(10,s.n_features)); truth=simulate(s,x,r.deployed_params)
    for sr in r.basis:
        assert np.max(np.abs(evaluate_surrogate(sr,x)-truth[sr['readout']]))<1e-11
