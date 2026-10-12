const $ = (id) => document.getElementById(id);
let algorithmMetadata = [];
let datasetSchema = [];
let modelSchema = [];
let datasetReady = false;
let testedConfig = null;
let cleaningOperations = [];
let stagedCleaningOperations = [];

const notice = (message, error = false) => {
  const box = $('notice'); box.textContent = message; box.classList.remove('hidden');
  box.style.background = error ? '#fff0ed' : '#eaf5df';
  box.style.color = error ? '#9b4436' : '#45633a';
};
async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options); const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function renderDataset(data) {
  datasetSchema = data.schema; modelSchema = data.model_schema || data.schema; datasetReady = true; renderModelChoices(); syncBuilderControls();
  $('data-tools').classList.remove('hidden');
  cleaningOperations = data.cleaning_operations || cleaningOperations;
  stagedCleaningOperations = [...cleaningOperations];
  renderCleaningFields(); renderCleaningList(); renderEdaFields();
  const vizSelect = $('viz-columns'); vizSelect.innerHTML = datasetSchema.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} · ${escapeHtml(c.kind)}</option>`).join('');
  $('remove-dataset').classList.remove('hidden');
  const uploadedOption = document.querySelector('#play-source option[value="uploaded"]');
  if (uploadedOption) uploadedOption.disabled = false;
  renderPlayControls();
  const target = $('dataset-result'); target.classList.remove('hidden');
  target.innerHTML = `<div class="stats"><div class="stat">${data.rows.toLocaleString()} rows</div><div class="stat">${data.columns} columns</div><div class="stat">${data.summary.duplicates} duplicate rows</div><div class="stat">${data.summary.constant_columns?.length || 0} constant columns</div></div><h3>Column overview</h3><table><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>${data.schema.map(c => `<tr><td>${escapeHtml(c.name)}</td><td>${escapeHtml(c.kind)}</td><td>${c.missing} (${c.missing_percent}%)</td><td>${c.unique}</td></tr>`).join('')}</tbody></table><h3>Numeric statistics and correlations</h3><pre>${escapeHtml(JSON.stringify({numeric: data.summary.numeric_summary, correlations: data.summary.correlations, constant_columns: data.summary.constant_columns}, null, 2))}</pre><h3>Most frequent categorical values</h3><pre>${escapeHtml(JSON.stringify(data.summary.categorical_summary, null, 2))}</pre>`;
}
function columnOptions(kind = null) { return datasetSchema.filter(c => !kind || c.kind === kind).map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} · ${escapeHtml(c.kind)}</option>`).join(''); }
function selectedValues(id) { return [...($(id)?.selectedOptions || [])].map(option => option.value); }
function renderCleaningFields() {
  const type = $('clean-type').value, host = $('clean-fields');
  if (type === 'impute' || type === 'drop_missing_rows' || type === 'drop_missing_columns') {
    host.innerHTML = `<label>Columns <select id="clean-columns" multiple size="4">${columnOptions()}</select></label>${type === 'impute' ? '<label>Fill strategy<select id="clean-strategy"><option value="mean">Mean (numeric only)</option><option value="median">Median (numeric only)</option><option value="mode">Most common value</option></select></label>' : ''}`;
  } else if (type === 'outliers') {
    host.innerHTML = `<label>Numeric columns<select id="clean-columns" multiple size="4">${columnOptions('numeric')}</select></label><p class="hint">The IQR rule flags possible anomalies. Detection never removes or edits rows.</p>`;
  } else if (type === 'convert') {
    host.innerHTML = `<label>Column<select id="clean-column">${columnOptions()}</select></label><label>Convert to<select id="clean-target"><option value="numeric">Numeric</option><option value="category">Categorical</option><option value="boolean">Boolean</option><option value="datetime">Date/time</option></select></label>`;
  } else if (type === 'encode') {
    host.innerHTML = `<label>Columns to encode<select id="clean-columns" multiple size="4">${columnOptions()}</select></label><label>Encoding<select id="clean-method"><option value="onehot">One-hot · nominal categories</option><option value="ordinal">Ordinal · genuinely ordered categories</option></select></label><label>Order for one selected ordinal column<input id="clean-category-order" placeholder="low, medium, high"></label><label class="check"><input type="checkbox" id="clean-ordered"> I confirm this category has a meaningful order</label><p class="hint">One-hot is the safe choice for labels such as color or city. Ordinal encoding needs the actual order, for example low, medium, high. Predictive model pipelines encode categories using training data only.</p>`;
  } else host.innerHTML = '<p class="hint">Duplicate removal keeps the first copy of each identical row.</p>';
}
function renderCleaningList() {
  $('clean-list').innerHTML = stagedCleaningOperations.length ? `<strong>Selected steps</strong>${stagedCleaningOperations.map((op, i) => `<div class="workflow-result-item"><span>${i + 1}. ${escapeHtml(op.type.replaceAll('_', ' '))}</span> ${escapeHtml((op.columns || (op.column ? [op.column] : [])).join(', '))}</div>`).join('')}` : '<p class="hint">No cleaning steps selected.</p>';
}
function collectCleaningOperation() {
  const type = $('clean-type').value, columns = selectedValues('clean-columns');
  if (type === 'impute') return {type, columns, strategies: Object.fromEntries(columns.map(c => [c, $('clean-strategy').value]))};
  if (type === 'drop_missing_rows' || type === 'drop_missing_columns') return {type, columns};
  if (type === 'outliers') return {type, columns};
  if (type === 'convert') return {type, column: $('clean-column').value, target: $('clean-target').value};
  if (type === 'encode') {
    const method = $('clean-method').value, categoryOrder = $('clean-category-order').value.split(',').map(v => v.trim()).filter(Boolean);
    if (method === 'ordinal' && (! $('clean-ordered').checked || columns.length !== 1 || categoryOrder.length < 2)) throw new Error('Ordinal encoding needs one selected column, its ordered category values, and your confirmation.');
    return {type, columns, method, ordered: $('clean-ordered').checked, category_order: method === 'ordinal' ? categoryOrder : null};
  }
  return {type: 'drop_duplicates'};
}
function renderEdaFields() {
  const type = $('eda-type').value, host = $('eda-fields');
  const kind = ['histogram', 'boxplot', 'scatter', 'correlation'].includes(type) ? 'numeric' : 'categorical';
  if (type === 'scatter') host.innerHTML = `<label>Numeric columns (choose two)<select id="eda-columns" multiple size="5">${columnOptions('numeric')}</select></label>`;
  else if (type === 'correlation') host.innerHTML = `<label>Numeric columns (optional; empty means all)<select id="eda-columns" multiple size="5">${columnOptions('numeric')}</select></label>`;
  else host.innerHTML = `<label>${kind === 'numeric' ? 'Numeric column' : 'Category column'}<select id="eda-columns">${columnOptions(kind)}</select></label>`;
}
function workflowConfig() {
  const algorithms = [...document.querySelectorAll('[data-algorithm]:checked')].map(input => {
    const spec = algorithmMetadata.find(item => item.id === input.dataset.algorithm);
    const selector = document.querySelector(`[data-target-for="${input.dataset.algorithm}"]`);
    const id = input.dataset.algorithm, features = selectedValues(`features-${id}`);
    const parameters = {random_state: Number($(`seed-${id}`)?.value || 42)};
    if (features.length) parameters.feature_columns = features;
    if (spec?.category !== 'clustering') {
      parameters.test_size = Number($(`split-${id}`)?.value || .2);
      parameters.scaler = $(`scaler-${id}`)?.value || (['logistic_regression', 'knn'].includes(id) ? 'standard' : 'none');
    } else {
      parameters.n_clusters = Number($(`k-${id}`)?.value || 3);
      parameters.scaler = $(`scaler-${id}`)?.value || 'standard';
    }
    if (id === 'decision_tree') parameters.max_depth = Number($(`depth-${id}`)?.value || 4);
    if (id === 'knn') parameters.n_neighbors = Number($(`neighbors-${id}`)?.value || 5);
    if (id === 'logistic_regression') parameters.max_iter = Number($(`iterations-${id}`)?.value || 1000);
    return {id, target: spec?.supervised ? (selector?.value || null) : null, parameters};
  });
  const includePlayground = $('include-playground').checked;
  const visualizations = $('include-viz').checked;
  return {visualizations, visualization_types: visualizations ? [...document.querySelectorAll('[data-viz-type]:checked')].map(input => input.dataset.vizType) : [],
    visualization_columns: visualizations ? selectedValues('viz-columns') : [], drop_duplicates: $('include-dupes').checked,
    cleaning_operations: cleaningOperations, include_playground: includePlayground,
    playground: includePlayground ? experimentConfig() : null, algorithms};
}
function selectionKey(config) { return JSON.stringify(config); }
function invalidateTest() {
  testedConfig = null; $('workflow-results').classList.add('hidden'); syncBuilderControls();
}
function syncBuilderControls() {
  const selected = [...document.querySelectorAll('[data-algorithm]:checked')];
  document.querySelectorAll('[data-target-for]').forEach(select => {
    const checkbox = document.querySelector(`[data-algorithm="${select.dataset.targetFor}"]`);
    select.disabled = !datasetReady || !checkbox?.checked;
    select.closest('.algorithm-choice')?.classList.toggle('selected', Boolean(checkbox?.checked));
  });
  const targetsReady = selected.every(input => {
    const spec = algorithmMetadata.find(item => item.id === input.dataset.algorithm);
    if (!spec?.supervised) return true;
    return Boolean(document.querySelector(`[data-target-for="${input.dataset.algorithm}"]`)?.value);
  });
  const testButton = $('test-workflow'), generateButton = $('generate');
  const hasOtherSections = $('include-playground').checked || $('include-viz').checked || $('include-dupes').checked || cleaningOperations.length > 0;
  if (testButton) testButton.disabled = !datasetReady || (selected.length === 0 && !hasOtherSections) || !targetsReady;
  if (generateButton) generateButton.disabled = !testedConfig || testedConfig !== selectionKey(workflowConfig());
}
function renderModelChoices() {
  const host = $('model-options');
  if (!algorithmMetadata.length) return;
  host.innerHTML = algorithmMetadata.map(item => {
    const targetControl = item.supervised
      ? `<label>Target<select data-target-for="${item.id}" aria-label="Target column for ${escapeHtml(item.name)}" disabled><option value="">Choose target column…</option>${modelSchema.filter(c => item.id !== 'linear_regression' || c.kind === 'numeric').map(column => `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)} · ${escapeHtml(column.kind)}</option>`).join('')}</select></label>`
      : '<span class="target-note">No target needed · groups data</span>';
    const available = item.id === 'kmeans' ? modelSchema.filter(c => c.kind === 'numeric') : modelSchema;
    const featureControl = `<label>Features<select id="features-${item.id}" multiple size="3" aria-label="Optional features for ${escapeHtml(item.name)}">${available.map(c=>`<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('')}</select><small>Leave blank to use all eligible columns.</small></label>`;
    const splitAndSeed = item.supervised ? `<label>Test fraction<input id="split-${item.id}" type="number" min="0.05" max="0.5" step="0.05" value="0.2"></label><label>Scale numeric features<select id="scaler-${item.id}"><option value="${['logistic_regression','knn'].includes(item.id)?'standard':'none'}">${['logistic_regression','knn'].includes(item.id)?'Standardize (recommended)':'No scaling (recommended)'}</option><option value="standard">StandardScaler</option><option value="minmax">MinMaxScaler</option><option value="none">No scaling</option></select></label>` : `<label>Clusters K<input id="k-${item.id}" type="number" min="2" max="12" value="3"></label><label>Scale for distance<select id="scaler-${item.id}"><option value="standard">StandardScaler</option><option value="minmax">MinMaxScaler</option><option value="none">No scaling</option></select></label>`;
    const modelParams = `${splitAndSeed}${item.id === 'decision_tree' ? `<label>Maximum depth<input id="depth-${item.id}" type="number" min="1" max="30" value="4"></label>` : ''}${item.id === 'knn' ? `<label>Neighbors K<input id="neighbors-${item.id}" type="number" min="1" max="100" value="5"></label>` : ''}${item.id === 'logistic_regression' ? `<label>Max iterations<input id="iterations-${item.id}" type="number" min="100" max="5000" step="100" value="1000"></label>` : ''}<label>Random seed<input id="seed-${item.id}" type="number" value="42"></label>`;
    return `<div class="algorithm-choice ${item.id === 'kmeans' ? 'selected' : ''}"><label class="choice-name"><input type="checkbox" data-algorithm="${item.id}" ${item.id === 'kmeans' ? 'checked' : ''} ${datasetReady ? '' : 'disabled'}> ${escapeHtml(item.name)}</label><div class="algorithm-settings">${targetControl}${featureControl}<div class="algorithm-parameters">${modelParams}</div></div></div>`;
  }).join('');
  syncBuilderControls();
}
async function upload(file) {
  if (!file) return;
  datasetReady = false; datasetSchema = []; testedConfig = null; syncBuilderControls();
  cleaningOperations = []; stagedCleaningOperations = [];
  const form = new FormData(); form.append('file', file);
  try {
    notice('Reading your dataset locally…');
    const data = await jsonFetch('/api/upload', {method: 'POST', body: form});
    renderDataset(data); notice(`Ready: ${file.name}`);
  } catch (error) { notice(error.message, true); }
}
$('file').addEventListener('change', event => upload(event.target.files[0]));
const drop = $('dropzone');
drop.addEventListener('dragover', event => { event.preventDefault(); drop.style.borderColor = '#809c52'; });
drop.addEventListener('dragleave', () => drop.style.borderColor = '');
drop.addEventListener('drop', event => { event.preventDefault(); drop.style.borderColor = ''; upload(event.dataTransfer.files[0]); });
async function loadAlgorithms() {
  try {
    algorithmMetadata = await jsonFetch('/api/algorithms');
    $('algorithm-grid').innerHTML = algorithmMetadata.map(item => `<article class="algorithm-card"><div class="category">${escapeHtml(item.category)}${item.supervised ? ' · supervised' : ' · unsupervised'}</div><h3>${escapeHtml(item.name)}</h3><p>${escapeHtml(item.description)}</p></article>`).join('');
    renderModelChoices();
  } catch (error) { notice(error.message, true); }
}
loadAlgorithms();
$('model-options').addEventListener('change', event => {
  if (event.target.matches('[data-algorithm]')) {
    const selector = document.querySelector(`[data-target-for="${event.target.dataset.algorithm}"]`);
    if (selector) selector.disabled = !event.target.checked;
    event.target.closest('.algorithm-choice')?.classList.toggle('selected', event.target.checked);
  }
  invalidateTest();
});
['include-viz', 'include-dupes', 'include-playground'].forEach(id => $(id).addEventListener('change', () => {
  if ($('visualization-options')) $('visualization-options').classList.toggle('hidden', !$('include-viz').checked);
  invalidateTest();
}));
document.querySelectorAll('[data-viz-type]').forEach(input => input.addEventListener('change', invalidateTest));
$('viz-columns').addEventListener('change', invalidateTest);
document.querySelector('#playground').addEventListener('change', () => { if ($('include-playground').checked) invalidateTest(); });
const testButton = $('test-workflow');
if (testButton) testButton.addEventListener('click', async () => {
  const config = workflowConfig();
  try {
    testButton.disabled = true; testButton.textContent = 'Testing…';
    const result = await jsonFetch('/api/workflow/test', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config)});
    const box = $('workflow-results'); box.classList.remove('hidden');
    box.innerHTML = `<h4>${result.passed ? 'All selected experiments passed' : 'Some selections need attention'}</h4>${result.results.map(item => {
      const metrics = item.metrics ? Object.entries(item.metrics).filter(([, value]) => value !== null).map(([key, value]) => `${escapeHtml(key)}: ${typeof value === 'number' ? value.toFixed(3) : escapeHtml(value)}`).join(' · ') : '';
      const sectionNames = {playground:'Playground experiment', cleaning:'Cleaning steps', visualizations:'EDA charts'};
      const name = algorithmMetadata.find(a => a.id === item.id)?.name || sectionNames[item.id] || item.id;
      return `<div class="workflow-result-item ${item.ok ? 'ok' : 'fail'}"><strong>${item.ok ? '✓' : '!' } ${escapeHtml(name)}</strong>${escapeHtml(item.message)}${metrics ? `<div class="workflow-metrics">${metrics}</div>` : ''}</div>`;
    }).join('')}`;
    testedConfig = result.passed ? selectionKey(config) : null;
    syncBuilderControls();
    if (result.passed) notice('Selection tested successfully. You can now generate the notebook.');
  } catch (error) { notice(error.message, true); }
  finally { testButton.textContent = 'Test selections ✓'; syncBuilderControls(); }
});
const generateButton = $('generate');
if (generateButton) generateButton.addEventListener('click', async () => {
  const config = workflowConfig();
  try {
    const response = await fetch('/api/notebook', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config)});
    if (!response.ok) { const error = await response.json(); throw new Error(error.error || 'Could not generate notebook.'); }
    const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a');
    link.href = url; link.download = 'dataforge_analysis.ipynb'; link.click(); URL.revokeObjectURL(url);
    notice('Notebook downloaded with your tested target selections.');
  } catch (error) { notice(error.message, true); }
});
const plotColors = ['#709746','#e3a656','#678ec1','#ce7180','#907bb8','#55a99a','#bf8a62','#81934b'];
function field(label, id, options, value = '') {
  return `<label class="play-controls-field">${label}<select id="${id}">${options.map(option => `<option value="${escapeHtml(option.value)}" ${String(option.value)===String(value)?'selected':''}>${escapeHtml(option.label)}</option>`).join('')}</select></label>`;
}
function rangeField(label, id, min, max, value, step=1) {
  return `<label class="play-controls-field">${label}<div class="range-line"><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output id="${id}-value">${value}</output></div></label>`;
}
function renderPlayControls() {
  const algorithm = $('play-algorithm').value, source = $('play-source').value, host = $('play-controls');
  let markup = '';
  if (source === 'sample') {
    const shapes = algorithm === 'linear_regression' ? [{value:'line',label:'Noisy linear trend'}]
      : [{value:'blobs',label:'Separated groups'},{value:'moons',label:'Two moons'},{value:'circles',label:'Concentric circles'}];
    markup += field('Sample pattern', 'play-shape', shapes, algorithm === 'linear_regression' ? 'line' : 'blobs');
    markup += field('Number of examples', 'play-samples', [{value:200,label:'200'},{value:300,label:'300'},{value:500,label:'500'}], 300);
  } else {
    const numeric = modelSchema.filter(column => column.kind === 'numeric');
    const featureOptions = numeric.map(column => ({value:column.name,label:column.name}));
    if (algorithm === 'linear_regression') {
      markup += field('Input feature (number)', 'play-feature-x', featureOptions, featureOptions[0]?.value);
    } else {
      markup += field('Horizontal feature (number)', 'play-feature-x', featureOptions, featureOptions[0]?.value);
      markup += field('Vertical feature (number)', 'play-feature-y', featureOptions, featureOptions[1]?.value ?? featureOptions[0]?.value);
    }
    if (algorithm !== 'kmeans') {
      const targets = modelSchema.filter(column => algorithm === 'linear_regression' ? column.kind === 'numeric' : true);
      markup += field(algorithm === 'linear_regression' ? 'Value to predict (number)' : 'Category to predict', 'play-target', targets.map(column => ({value:column.name,label:`${column.name} · ${column.kind}`})), '');
    }
    if (!numeric.length) markup += '<p class="hint">This dataset has no numeric columns to plot. Choose a different dataset.</p>';
  }
  if (algorithm === 'kmeans') {
    markup += rangeField('Clusters (K)', 'play-clusters', 2, 12, 3);
    markup += '<label class="play-controls-check"><input id="play-scale" type="checkbox" checked> Standardize feature scale</label>';
  } else if (algorithm === 'linear_regression') {
    markup += rangeField('Noise in sample data', 'play-noise', 0, 40, 15);
  } else if (algorithm === 'decision_tree') {
    markup += rangeField('Maximum tree depth', 'play-depth', 1, 12, 4);
  } else if (algorithm === 'knn') {
    markup += rangeField('Neighbors (K)', 'play-neighbors', 1, 25, 5);
  }
  markup += '<label class="play-controls-field">Random seed<input id="play-seed" type="number" value="42"></label>';
  host.innerHTML = markup;
  $('play-regenerate').classList.toggle('hidden', source !== 'sample');
  host.querySelectorAll('select').forEach(select => select.addEventListener('change', () => {
    $('play-notice').textContent = ''; $('play-notice').dataset.error = ''; updatePlaygroundReadiness();
  }));
  updatePlaygroundReadiness();
  host.querySelectorAll('input[type=range]').forEach(input => input.addEventListener('input', () => {
    const output = $(`${input.id}-value`); if (output) output.value = input.value;
  }));
  $('play-legend').innerHTML = '';
  $('silhouette').textContent = 'Run an experiment to see results.'; $('cluster-sizes').textContent = '';
}
$('play-algorithm').addEventListener('change', renderPlayControls);
$('play-source').addEventListener('change', renderPlayControls);
$('clean-type').addEventListener('change', renderCleaningFields);
$('eda-type').addEventListener('change', renderEdaFields);
$('clean-add').addEventListener('click', () => {
  try {
    const operation = collectCleaningOperation();
    if (['impute', 'drop_missing_rows', 'drop_missing_columns', 'encode', 'outliers'].includes(operation.type) && !operation.columns.length) throw new Error('Choose at least one column for this operation.');
    stagedCleaningOperations.push(operation); renderCleaningList(); $('clean-notice').textContent = '';
  } catch (error) { $('clean-notice').textContent = error.message; }
});
$('clean-preview').addEventListener('click', async () => {
  try {
    const result = await jsonFetch('/api/cleaning/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(stagedCleaningOperations)});
    $('clean-notice').textContent = `Preview: ${result.before.rows} → ${result.after.rows} rows, ${result.before.columns} → ${result.after.columns} columns, ${result.before.missing} → ${result.after.missing} missing values. ${result.reports.map(r => r.message + (r.details ? ' ' + r.details.map(d => `${d.column}: ${d.count} potential outliers`).join(', ') : '')).join(' ')}`;
  } catch(error) { $('clean-notice').textContent = error.message; }
});
$('clean-apply').addEventListener('click', async () => {
  try {
    const result = await jsonFetch('/api/cleaning/apply', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(stagedCleaningOperations)});
    cleaningOperations = [...stagedCleaningOperations]; testedConfig = null;
    renderDataset({...result, cleaning_operations: cleaningOperations}); renderCleaningList(); syncBuilderControls();
    $('clean-notice').textContent = result.reports.map(r => r.message).join(' ') || 'Dataset reset to the uploaded original.';
  } catch(error) { $('clean-notice').textContent = error.message; }
});
$('clean-reset').addEventListener('click', async () => {
  stagedCleaningOperations = []; renderCleaningList();
  try {
    const result = await jsonFetch('/api/cleaning/apply', {method:'POST', headers:{'Content-Type':'application/json'}, body:'[]'});
    cleaningOperations = []; testedConfig = null; renderDataset({...result, cleaning_operations: []});
    $('clean-notice').textContent = 'Cleaning reset. Your uploaded original remains unchanged.';
  } catch(error) { $('clean-notice').textContent = error.message; }
});
$('eda-run').addEventListener('click', async () => {
  try {
    const input = $('eda-columns'), columns = input.multiple ? selectedValues('eda-columns') : [input.value];
    const data = await jsonFetch('/api/eda/chart', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({type:$('eda-type').value, columns})});
    drawEdaChart(data); $('eda-notice').textContent = data.message || (data.truncated ? 'Showing the most frequent 20 categories.' : '');
  } catch(error) { $('eda-notice').textContent = error.message; }
});
function drawEdaChart(data) {
  const canvas=$('eda-chart'), width=canvas.clientWidth||550, height=250, ratio=window.devicePixelRatio||1;
  canvas.width=width*ratio; canvas.height=height*ratio; const ctx=canvas.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,width,height);
  const left=48, top=25, bottom=height-34, right=width-12;
  canvas.title = 'Hover over a point or bar to see its values.'; canvas.dataset.edaType = data.type; canvas._edaPayload = data;
  ctx.fillStyle='#3d4b47'; ctx.font='12px sans-serif'; ctx.textAlign='center'; ctx.fillText(data.type==='scatter'?`${data.x} vs ${data.y}`:`${data.type}: ${data.column||'selected numeric columns'}`,width/2,15);
  if(data.type==='scatter') {
    const xs=data.points.map(p=>p.x), ys=data.points.map(p=>p.y), xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
    ctx.fillStyle='#709746'; data.points.forEach(p=>{ctx.beginPath();ctx.arc(left+(p.x-xmin)/(xmax-xmin||1)*(right-left),bottom-(p.y-ymin)/(ymax-ymin||1)*(bottom-top),2.5,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle='#687675';ctx.fillText(data.x,width/2,height-5);ctx.save();ctx.translate(11,height/2);ctx.rotate(-Math.PI/2);ctx.fillText(data.y,0,0);ctx.restore();
    canvas.onmousemove=event=>{const rect=canvas.getBoundingClientRect(),mx=event.clientX-rect.left,my=event.clientY-rect.top;let best=null,distance=Infinity;data.points.forEach(p=>{const dx=left+(p.x-xmin)/(xmax-xmin||1)*(right-left)-mx,dy=bottom-(p.y-ymin)/(ymax-ymin||1)*(bottom-top)-my,d=Math.hypot(dx,dy);if(d<distance){distance=d;best=p;}});canvas.title=best&&distance<14?`${data.x}: ${best.x.toPrecision(5)}, ${data.y}: ${best.y.toPrecision(5)}`:'Hover near a point to inspect its values.';};return;
  }
  if(data.type==='correlation') {
    const n=data.labels.length, cell=Math.min((right-left)/n,(bottom-top)/n);
    data.matrix.forEach((row,i)=>row.forEach((v,j)=>{const t=Math.max(0,Math.min(1,(v+1)/2));ctx.fillStyle=`rgb(${Math.round(238-110*t)},${Math.round(242-75*t)},${Math.round(233-40*t)})`;ctx.fillRect(left+j*cell,top+i*cell,cell-1,cell-1);ctx.fillStyle='#33413b';ctx.font='9px sans-serif';ctx.fillText(v.toFixed(2),left+(j+.5)*cell,top+(i+.55)*cell);}));
    ctx.textAlign='right';ctx.font='9px sans-serif';data.labels.forEach((label,i)=>ctx.fillText(label,left-4,top+(i+.65)*cell));canvas.onmousemove=event=>{const rect=canvas.getBoundingClientRect(),j=Math.floor((event.clientX-rect.left-left)/cell),i=Math.floor((event.clientY-rect.top-top)/cell);canvas.title=i>=0&&j>=0&&i<data.labels.length&&j<data.labels.length?`${data.labels[i]} × ${data.labels[j]}: correlation ${data.matrix[i][j]}`:'Hover over a heatmap cell.';};return;
  }
  if(data.type==='boxplot') {
    const values=[data.minimum,data.q1,data.median,data.q3,data.maximum], min=values[0],max=values[4], x=v=>left+(v-min)/(max-min||1)*(right-left), mid=(top+bottom)/2;
    ctx.strokeStyle='#709746';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(min),mid);ctx.lineTo(x(max),mid);ctx.moveTo(x(min),mid-9);ctx.lineTo(x(min),mid+9);ctx.moveTo(x(max),mid-9);ctx.lineTo(x(max),mid+9);ctx.stroke();ctx.fillStyle='#e5eddc';ctx.fillRect(x(data.q1),mid-22,x(data.q3)-x(data.q1),44);ctx.strokeStyle='#526844';ctx.strokeRect(x(data.q1),mid-22,x(data.q3)-x(data.q1),44);ctx.beginPath();ctx.moveTo(x(data.median),mid-22);ctx.lineTo(x(data.median),mid+22);ctx.stroke();ctx.fillStyle='#687675';ctx.font='10px sans-serif';ctx.textAlign='center';['min','Q1','median','Q3','max'].forEach((label,i)=>ctx.fillText(label,x(values[i]),bottom+16));canvas.onmousemove=event=>{const rect=canvas.getBoundingClientRect(),vx=min+(event.clientX-rect.left-left)/(right-left)*(max-min||1),names=['Minimum','Q1','Median','Q3','Maximum'];let ix=values.reduce((best,v,i)=>Math.abs(v-vx)<Math.abs(values[best]-vx)?i:best,0);canvas.title=`${names[ix]}: ${values[ix].toPrecision(5)}`;};return;
  }
  const values=data.values||[], labels=data.labels||[], max=Math.max(...values,1), gap=5, bw=(right-left)/Math.max(values.length,1);
  values.forEach((v,i)=>{const bh=(v/max)*(bottom-top);ctx.fillStyle='#82a75a';ctx.fillRect(left+i*bw+gap, bottom-bh, Math.max(1,bw-gap*2), bh);ctx.fillStyle='#65716c';ctx.font='9px sans-serif';ctx.textAlign='center';if(values.length<15)ctx.fillText(labels[i],left+(i+.5)*bw,bottom+13);});
  ctx.save();ctx.translate(12,height/2);ctx.rotate(-Math.PI/2);ctx.fillText('Count',0,0);ctx.restore();
  ctx.fillStyle='#687675';ctx.font='9px sans-serif';ctx.textAlign='center';ctx.fillText(data.column||'Category',width/2,height-3);
  canvas.onmousemove=event=>{const rect=canvas.getBoundingClientRect(),index=Math.floor((event.clientX-rect.left-left)/bw);canvas.title=index>=0&&index<labels.length?`${labels[index]}: ${values[index]} rows`:'Hover over a bar for its count.';};
}
function experimentConfig() {
  const algorithm = $('play-algorithm').value, source = $('play-source').value;
  return {algorithm, source, shape: $('play-shape')?.value || 'blobs',
    n_samples: Number($('play-samples')?.value || 300),
    n_clusters: Number($('play-clusters')?.value || 3),
    scale: $('play-scale')?.checked ?? true,
    noise: Number($('play-noise')?.value || 15),
    max_depth: Number($('play-depth')?.value || 4),
    n_neighbors: Number($('play-neighbors')?.value || 5),
    random_state: Number($('play-seed')?.value || 42),
    feature_x: $('play-feature-x')?.value || null,
    feature_y: $('play-feature-y')?.value || null,
    target: $('play-target')?.value || null};
}
function updatePlaygroundReadiness() {
  const button = $('run-playground'), noticeBox = $('play-notice'); if (!button || !noticeBox) return;
  const config = experimentConfig(); let message = '';
  if (config.source === 'uploaded') {
    if (!datasetReady) message = 'Upload a CSV first, or choose built-in sample data.';
    else if (!config.feature_x) message = 'This CSV needs at least one numeric column to use as a feature.';
    else if (config.algorithm !== 'linear_regression' && !config.feature_y) message = 'Choose a second numeric column for the chart.';
    else if (config.algorithm !== 'linear_regression' && config.feature_x === config.feature_y) message = 'Choose two different numeric feature columns.';
    else if (config.algorithm !== 'kmeans' && !config.target) message = 'Choose the column you want the model to predict.';
    else if (config.algorithm !== 'kmeans' && [config.feature_x, config.feature_y].includes(config.target)) message = 'Choose a target column different from the feature columns.';
  }
  button.disabled = Boolean(message);
  if (message) { noticeBox.textContent = message; noticeBox.style.color = '#9b4436'; noticeBox.dataset.error = 'validation'; }
  else if (noticeBox.dataset.error === 'validation') { noticeBox.textContent = ''; noticeBox.dataset.error = ''; }
}
function drawExperiment(data) {
  $('prediction-chart').classList.toggle('hidden', !data.actual_vs_predicted);
  $('elbow-chart').classList.toggle('hidden', !data.elbow);
  const canvas=$('plot'), ratio=window.devicePixelRatio||1, width=canvas.clientWidth||650, height=270;
  canvas.width=width*ratio;canvas.height=height*ratio;const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  const all=[...data.points,...(data.boundary||[]),...(data.curve||[]),...(data.centroids||[]).map(point=>({x:point[0],y:point[1]}))];
  if(!all.length)return;
  const xs=all.map(point=>point.x),ys=all.map(point=>point.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),pad=22;
  const px=x=>pad+(x-xmin)/(xmax-xmin||1)*(width-pad*2),py=y=>height-pad-(y-ymin)/(ymax-ymin||1)*(height-pad*2);
  ctx.strokeStyle='#edf0eb';ctx.lineWidth=1;for(let i=1;i<5;i++){ctx.beginPath();ctx.moveTo(pad,height*i/5);ctx.lineTo(width-pad,height*i/5);ctx.stroke();}
  const classes=data.classes||[],colorFor=label=>plotColors[Math.max(0,classes.indexOf(String(label)))%plotColors.length];
  if(data.boundary){const w=Math.max(3,(width-pad*2)/35),h=Math.max(3,(height-pad*2)/35);ctx.globalAlpha=.2;data.boundary.forEach(cell=>{ctx.fillStyle=colorFor(cell.label);ctx.fillRect(px(cell.x)-w/2,py(cell.y)-h/2,w+1,h+1);});ctx.globalAlpha=1;}
  data.points.forEach(point=>{ctx.beginPath();ctx.fillStyle=point.label===undefined?'#729849':colorFor(point.label);ctx.globalAlpha=.83;ctx.arc(px(point.x),py(point.y),3.7,0,Math.PI*2);ctx.fill();});ctx.globalAlpha=1;
  if(data.curve){ctx.beginPath();ctx.strokeStyle='#d1794b';ctx.lineWidth=3;data.curve.forEach((point,index)=>index?ctx.lineTo(px(point.x),py(point.y)):ctx.moveTo(px(point.x),py(point.y)));ctx.stroke();}
  (data.centroids||[]).forEach(point=>{ctx.beginPath();ctx.strokeStyle='#162728';ctx.lineWidth=2.5;ctx.moveTo(px(point[0])-6,py(point[1])-6);ctx.lineTo(px(point[0])+6,py(point[1])+6);ctx.moveTo(px(point[0])+6,py(point[1])-6);ctx.lineTo(px(point[0])-6,py(point[1])+6);ctx.stroke();});
  ctx.fillStyle='#687675';ctx.font='10px sans-serif';ctx.textAlign='center';ctx.fillText(data.axis_x||'Feature 1',width/2,height-3);
  ctx.save();ctx.translate(9,height/2);ctx.rotate(-Math.PI/2);ctx.fillText(data.axis_y||'Feature 2',0,0);ctx.restore();
  $('play-legend').innerHTML=classes.map(label=>`<span class="legend-item"><i class="legend-dot" style="background:${colorFor(label)}"></i>${escapeHtml(data.algorithm==='kmeans'?`Cluster ${label}`:label)}</span>`).join('');
  if(data.elbow) drawElbow(data.elbow, data.elbow_sampled);
  if(data.actual_vs_predicted) drawPredictionComparison(data.actual_vs_predicted);
}
function drawElbow(values, sampled) {
  const canvas=$('elbow-chart'), width=canvas.clientWidth||650, height=190, ratio=window.devicePixelRatio||1;
  canvas.width=width*ratio;canvas.height=height*ratio;const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  const left=45,right=width-16,top=25,bottom=height-32,max=Math.max(...values.map(v=>v.inertia),1);
  ctx.strokeStyle='#df7960';ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{const x=left+i*(right-left)/Math.max(1,values.length-1),y=bottom-v.inertia/max*(bottom-top);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  values.forEach((v,i)=>{const x=left+i*(right-left)/Math.max(1,values.length-1),y=bottom-v.inertia/max*(bottom-top);ctx.fillStyle='#df7960';ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#65716c';ctx.font='9px sans-serif';ctx.textAlign='center';ctx.fillText(v.k,x,bottom+13);});
  ctx.fillStyle='#46544e';ctx.font='11px sans-serif';ctx.textAlign='center';ctx.fillText(`Elbow curve: within-cluster variation by K${sampled?' (2,000-row sample)':''}`,width/2,13);ctx.fillText('K (number of clusters)',width/2,height-3);
}
function drawPredictionComparison(values) {
  const canvas=$('prediction-chart'), width=canvas.clientWidth||650, height=190, ratio=window.devicePixelRatio||1;
  canvas.width=width*ratio;canvas.height=height*ratio;const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  const min=Math.min(...values.flatMap(v=>[v.actual,v.predicted])),max=Math.max(...values.flatMap(v=>[v.actual,v.predicted])),left=45,right=width-16,top=25,bottom=height-30,px=v=>left+(v-min)/(max-min||1)*(right-left),py=v=>bottom-(v-min)/(max-min||1)*(bottom-top);
  ctx.strokeStyle='#adb8a5';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px(min),py(min));ctx.lineTo(px(max),py(max));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#74964c';values.forEach(v=>{ctx.beginPath();ctx.arc(px(v.actual),py(v.predicted),3.5,0,Math.PI*2);ctx.fill();});ctx.fillStyle='#46544e';ctx.font='11px sans-serif';ctx.textAlign='center';ctx.fillText('Actual vs predicted (held-out test rows)',width/2,13);ctx.fillText('Actual value',width/2,height-3);ctx.save();ctx.translate(12,height/2);ctx.rotate(-Math.PI/2);ctx.fillText('Predicted value',0,0);ctx.restore();
}
function showExperimentMetrics(data) {
  const metrics=data.metrics||{};
  const labels={silhouette:'Silhouette',cluster_count:'Clusters',accuracy:'Accuracy',precision_weighted:'Precision',recall_weighted:'Recall',f1_weighted:'F1',mae:'MAE',rmse:'RMSE',r2:'R²'};
  $('silhouette').textContent=Object.entries(metrics).filter(([,value])=>value!==null).map(([key,value])=>`${labels[key]||key}: ${typeof value==='number'?value.toFixed(3):value}`).join(' · ')||'No metrics available';
  $('cluster-sizes').textContent=data.cluster_sizes?`Rows per cluster: ${Object.entries(data.cluster_sizes).map(([k,v])=>`${k}: ${v}`).join(', ')}`:data.rows_shown?`Showing ${data.rows_shown} rows`:'';
  const classification = ['logistic_regression','decision_tree','knn'].includes(data.algorithm);
  $('play-explanation').textContent=data.explanation + (classification ? ' Accuracy is the share correct; precision measures how often a predicted class is right; recall measures how many real examples of a class were found; F1 balances precision and recall.' : '');
}
async function runPlayground() {
  try {
    $('run-playground').disabled=true;$('run-playground').textContent='Running…';
    const data=await jsonFetch('/api/playground/experiment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(experimentConfig())});
    drawExperiment(data);showExperimentMetrics(data);
  } catch(error){const box=$('play-notice');box.textContent=error.message;box.style.color='#9b4436';box.dataset.error='server';} finally{$('run-playground').innerHTML='Run experiment <span>→</span>';updatePlaygroundReadiness();}
}
$('run-playground').addEventListener('click',runPlayground);
$('play-regenerate').addEventListener('click', () => {
  const seed = $('play-seed'); seed.value = String(Math.floor(Math.random() * 1000000)); runPlayground();
});
$('remove-dataset').addEventListener('click', async () => {
  try {
    await jsonFetch('/api/dataset', {method:'DELETE'});
    datasetReady = false; datasetSchema = []; modelSchema = []; cleaningOperations = []; stagedCleaningOperations = []; testedConfig = null;
    $('dataset-result').classList.add('hidden'); $('data-tools').classList.add('hidden'); $('remove-dataset').classList.add('hidden');
    const option=document.querySelector('#play-source option[value="uploaded"]'); if(option){option.disabled=true;if($('play-source').value==='uploaded')$('play-source').value='sample';}
    renderPlayControls(); renderModelChoices(); syncBuilderControls(); notice('Your uploaded dataset was removed from this browser session.');
  } catch(error) { notice(error.message,true); }
});
async function restoreDataset() {
  try { const data=await jsonFetch('/api/dataset');renderDataset(data); }
  catch (_) { renderPlayControls(); }
}
renderPlayControls();restoreDataset();
