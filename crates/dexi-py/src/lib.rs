//! Python bindings for dexi via PyO3.

use std::sync::Mutex;

use dexi::{RetargetingConfig, RetargetingType, SeqRetargeting};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn value_error(err: String) -> PyErr {
    PyValueError::new_err(err)
}

fn runtime_error(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

#[pyclass(name = "RetargetingConfig", skip_from_py_object)]
#[derive(Clone)]
struct PyRetargetingConfig {
    inner: RetargetingConfig,
}

#[pymethods]
impl PyRetargetingConfig {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RetargetingConfig::load_from_file(path).map_err(value_error)?,
        })
    }

    fn build(&self) -> PyResult<PySeqRetargeting> {
        let retargeting = self.inner.clone().build().map_err(runtime_error)?;
        Ok(PySeqRetargeting {
            inner: Mutex::new(retargeting),
        })
    }

    #[getter]
    fn type_(&self) -> &'static str {
        match self.inner.type_ {
            RetargetingType::Position => "position",
            RetargetingType::Vector => "vector",
            RetargetingType::DexPilot => "dexpilot",
        }
    }

    #[getter]
    fn urdf_path(&self) -> String {
        self.inner.urdf_path.clone()
    }

    #[getter]
    fn target_joint_names(&self) -> Option<Vec<String>> {
        self.inner.target_joint_names.clone()
    }
}

#[pyclass(name = "SeqRetargeting")]
struct PySeqRetargeting {
    inner: Mutex<SeqRetargeting>,
}

#[pymethods]
impl PySeqRetargeting {
    #[pyo3(signature = (ref_value, fixed_qpos=None))]
    fn retarget(&self, ref_value: Vec<f64>, fixed_qpos: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        Ok(guard.retarget(&ref_value, fixed_qpos.as_deref().unwrap_or(&[])))
    }

    fn reset(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        guard.reset();
        Ok(())
    }

    fn set_qpos(&self, robot_qpos: Vec<f64>) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        guard.set_qpos(&robot_qpos);
        Ok(())
    }

    #[pyo3(signature = (fixed_qpos=None))]
    fn get_qpos(&self, fixed_qpos: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        Ok(guard.get_qpos(fixed_qpos.as_deref()))
    }

    #[getter]
    fn joint_names(&self) -> PyResult<Vec<String>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        Ok(guard.optimizer.dof_joint_names())
    }

    #[getter]
    fn link_names(&self) -> PyResult<Vec<String>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        Ok(guard.optimizer.link_names())
    }

    fn link_positions(
        &self,
        robot_qpos: Vec<f64>,
        link_names: Vec<String>,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        guard
            .optimizer
            .link_positions(&robot_qpos, &link_names)
            .map(|points| points.into_iter().map(|p| (p[0], p[1], p[2])).collect())
            .map_err(value_error)
    }

    #[getter]
    fn fixed_dof(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        Ok(guard.optimizer.idx_pin2fixed().len())
    }

    #[getter]
    fn target_dof(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SeqRetargeting lock poisoned"))?;
        Ok(guard.optimizer.idx_pin2target().len())
    }
}

#[pyfunction]
fn load_from_file(path: &str) -> PyResult<PyRetargetingConfig> {
    PyRetargetingConfig::from_file(path)
}

/// Native Python extension module for dexi hand retargeting.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Native bindings for dexi-rs hand retargeting")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyRetargetingConfig>()?;
    m.add_class::<PySeqRetargeting>()?;
    m.add_function(wrap_pyfunction!(load_from_file, m)?)?;
    Ok(())
}
