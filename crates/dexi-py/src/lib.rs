//! Python bindings for dexi via PyO3.

use std::path::{Path, PathBuf};
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

/// Human-indices input: flat list, or nested rows flattened row-major
/// (matching the YAML parser's behavior).
#[derive(FromPyObject)]
enum IndicesInput {
    Flat(Vec<usize>),
    Nested(Vec<Vec<usize>>),
}

impl IndicesInput {
    fn flatten(self) -> Vec<usize> {
        match self {
            IndicesInput::Flat(v) => v,
            IndicesInput::Nested(rows) => rows.into_iter().flatten().collect(),
        }
    }
}

/// The bundled robots directory inside the installed dexi_rs package,
/// discovered from the native module's own location.
fn packaged_hands_dir(py: Python<'_>) -> Option<PathBuf> {
    let module = py.import("dexi_rs._native").ok()?;
    let file: String = module.getattr("__file__").ok()?.extract().ok()?;
    let dir = Path::new(&file)
        .parent()?
        .join("resources/assets/robots/hands");
    if dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}

#[pyclass(name = "RetargetingConfig", skip_from_py_object)]
#[derive(Clone)]
struct PyRetargetingConfig {
    inner: RetargetingConfig,
}

#[pymethods]
impl PyRetargetingConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        r#type,
        urdf_path,
        *,
        add_dummy_free_joint = false,
        target_link_human_indices = None,
        wrist_link_name = None,
        target_link_names = None,
        target_joint_names = None,
        target_origin_link_names = None,
        target_task_link_names = None,
        finger_tip_link_names = None,
        scaling_factor = 1.0,
        normal_delta = 4e-3,
        huber_delta = 2e-2,
        project_dist = 0.03,
        escape_dist = 0.05,
        has_joint_limits = true,
        ignore_mimic_joint = false,
        low_pass_alpha = 0.1,
        urdf_dir = None,
    ))]
    fn new(
        py: Python<'_>,
        r#type: &str,
        urdf_path: &str,
        add_dummy_free_joint: bool,
        target_link_human_indices: Option<IndicesInput>,
        wrist_link_name: Option<String>,
        target_link_names: Option<Vec<String>>,
        target_joint_names: Option<Vec<String>>,
        target_origin_link_names: Option<Vec<String>>,
        target_task_link_names: Option<Vec<String>>,
        finger_tip_link_names: Option<Vec<String>>,
        scaling_factor: f64,
        normal_delta: f64,
        huber_delta: f64,
        project_dist: f64,
        escape_dist: f64,
        has_joint_limits: bool,
        ignore_mimic_joint: bool,
        low_pass_alpha: f64,
        urdf_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let type_ = RetargetingType::from_str(r#type).map_err(value_error)?;
        let default_urdf_dir = urdf_dir
            .or_else(|| packaged_hands_dir(py))
            .unwrap_or_else(|| PathBuf::from("."));
        let inner = RetargetingConfig {
            type_,
            urdf_path: urdf_path.to_string(),
            add_dummy_free_joint,
            target_link_human_indices: target_link_human_indices.map(IndicesInput::flatten),
            wrist_link_name,
            target_link_names,
            target_joint_names,
            target_origin_link_names,
            target_task_link_names,
            finger_tip_link_names,
            scaling_factor,
            normal_delta,
            huber_delta,
            project_dist,
            escape_dist,
            has_joint_limits,
            ignore_mimic_joint,
            low_pass_alpha,
            default_urdf_dir,
        };
        inner.validate().map_err(value_error)?;
        Ok(Self { inner })
    }

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
    fn add_dummy_free_joint(&self) -> bool {
        self.inner.add_dummy_free_joint
    }

    #[getter]
    fn target_link_human_indices(&self) -> Option<Vec<usize>> {
        self.inner.target_link_human_indices.clone()
    }

    #[getter]
    fn wrist_link_name(&self) -> Option<String> {
        self.inner.wrist_link_name.clone()
    }

    #[getter]
    fn target_link_names(&self) -> Option<Vec<String>> {
        self.inner.target_link_names.clone()
    }

    #[getter]
    fn target_joint_names(&self) -> Option<Vec<String>> {
        self.inner.target_joint_names.clone()
    }

    #[getter]
    fn target_origin_link_names(&self) -> Option<Vec<String>> {
        self.inner.target_origin_link_names.clone()
    }

    #[getter]
    fn target_task_link_names(&self) -> Option<Vec<String>> {
        self.inner.target_task_link_names.clone()
    }

    #[getter]
    fn finger_tip_link_names(&self) -> Option<Vec<String>> {
        self.inner.finger_tip_link_names.clone()
    }

    #[getter]
    fn scaling_factor(&self) -> f64 {
        self.inner.scaling_factor
    }

    #[getter]
    fn normal_delta(&self) -> f64 {
        self.inner.normal_delta
    }

    #[getter]
    fn huber_delta(&self) -> f64 {
        self.inner.huber_delta
    }

    #[getter]
    fn project_dist(&self) -> f64 {
        self.inner.project_dist
    }

    #[getter]
    fn escape_dist(&self) -> f64 {
        self.inner.escape_dist
    }

    #[getter]
    fn has_joint_limits(&self) -> bool {
        self.inner.has_joint_limits
    }

    #[getter]
    fn ignore_mimic_joint(&self) -> bool {
        self.inner.ignore_mimic_joint
    }

    #[getter]
    fn low_pass_alpha(&self) -> f64 {
        self.inner.low_pass_alpha
    }

    #[getter]
    fn urdf_dir(&self) -> String {
        self.inner.default_urdf_dir.display().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RetargetingConfig(type='{}', urdf_path='{}', scaling_factor={})",
            self.type_(),
            self.inner.urdf_path,
            self.inner.scaling_factor
        )
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
