const state = {
    manifest: null,
    dataset: null,
    overview: null,
    scenesIndex: null,
    scene: null,
    sceneSummary: null,
    artifactScene: null,
    artifactSceneSummary: null,
    artifactMethod: null,
    detailCache: new Map(),
    tableFilters: new Map(),
    methodFilters: new Map(),
    openFilters: new Set(),
};

const CATEGORY_LABELS = {
    traj: 'Trajectory',
    auc: 'AUC',
    auc_micro: 'AUC Micro',
    auc_macro: 'AUC Macro',
    depth: 'Depth',
    points: 'Point Cloud',
};

const CATEGORY_ORDER = ['auc_micro', 'auc_macro', 'traj', 'auc', 'depth', 'points'];

window.addEventListener('DOMContentLoaded', init);

async function init() {
    bindControls();
    try {
        state.manifest = await fetchJson('data/manifest.json');
        initializeDatasetSelector();
        const params = new URLSearchParams(window.location.search);
        const requestedDataset = params.get('dataset');
        const dataset = findDataset(requestedDataset) || state.manifest.datasets[0];
        if (!dataset) {
            setStatus('No datasets in manifest.', 'error');
            return;
        }
        await loadDataset(dataset.id, {
            scene: params.get('scene'),
            artifactScene: params.get('artifact_scene') || params.get('scene'),
            artifactMethod: params.get('artifact_method') || params.get('method'),
        });
        setStatus(`Generated ${formatDate(state.manifest.generated_at)}`, 'ok');
    } catch (error) {
        console.error(error);
        setStatus(`Failed to load report: ${error.message}`, 'error');
    }
}

function bindControls() {
    document.getElementById('dataset-selector').addEventListener('change', async (event) => {
        await loadDataset(event.target.value);
    });
    document.getElementById('scene-breakdown-selector').addEventListener('change', async (event) => {
        await loadSceneBreakdown(event.target.value);
    });
    document.getElementById('artifact-scene-selector').addEventListener('change', async (event) => {
        await loadArtifactScene(event.target.value);
    });
    document.getElementById('artifact-method-selector').addEventListener('change', async (event) => {
        state.artifactMethod = event.target.value || null;
        await renderArtifactDetails();
        updateUrl();
    });
}

function initializeDatasetSelector() {
    const selector = document.getElementById('dataset-selector');
    selector.innerHTML = state.manifest.datasets.map((dataset) => (
        `<option value="${escapeAttr(dataset.id)}">${escapeHtml(dataset.label || dataset.id)}</option>`
    )).join('');
}

async function loadDataset(datasetId, requested = {}) {
    const dataset = findDataset(datasetId);
    if (!dataset) {
        setStatus(`Dataset not found: ${datasetId}`, 'error');
        return;
    }

    state.dataset = dataset;
    state.detailCache.clear();
    state.tableFilters.clear();
    state.methodFilters.clear();
    state.openFilters.clear();
    document.getElementById('dataset-selector').value = dataset.id;

    setStatus(`Loading ${dataset.id}...`, 'loading');
    const [overview, scenesIndex] = await Promise.all([
        fetchJson(dataset.overview_path),
        fetchJson(dataset.scenes_path),
    ]);
    state.overview = overview;
    state.scenesIndex = scenesIndex;

    renderDatasetOverview();
    initializeSceneSelectors();

    const sceneEntry = findScene(requested.scene) || scenesIndex.scenes[0];
    if (!sceneEntry) {
        state.scene = null;
        state.sceneSummary = null;
        state.artifactScene = null;
        state.artifactSceneSummary = null;
        state.artifactMethod = null;
        initializeArtifactMethodSelector();
        renderEmpty('scene-summary', 'No scenes available.');
        renderEmpty('artifact-details', 'No artifacts available.');
        updateUrl();
        setStatus(`Loaded ${dataset.id}`, 'ok');
        return;
    }

    const artifactScene = findScene(requested.artifactScene) || sceneEntry;
    await loadSceneBreakdown(sceneEntry.id, false);
    await loadArtifactScene(artifactScene.id, requested.artifactMethod, false);
    updateUrl();
    setStatus(`Loaded ${dataset.id}`, 'ok');
}

async function loadSceneBreakdown(sceneId, shouldUpdateUrl = true) {
    const scene = findScene(sceneId);
    if (!scene) {
        setStatus(`Scene not found: ${sceneId}`, 'error');
        return;
    }

    state.scene = scene;
    document.getElementById('scene-breakdown-selector').value = scene.id;
    state.sceneSummary = await fetchJson(scene.summary_path);
    renderSceneSummary();
    if (shouldUpdateUrl) updateUrl();
}

async function loadArtifactScene(sceneId, requestedMethod = null, shouldUpdateUrl = true) {
    const scene = findScene(sceneId);
    if (!scene) {
        setStatus(`Scene not found: ${sceneId}`, 'error');
        return;
    }

    state.artifactScene = scene;
    document.getElementById('artifact-scene-selector').value = scene.id;
    state.artifactSceneSummary = await fetchJson(scene.summary_path);

    const methods = currentArtifactMethods();
    state.artifactMethod = methods.includes(requestedMethod) ? requestedMethod : methods[0] || null;
    initializeArtifactMethodSelector();
    await renderArtifactDetails();
    if (shouldUpdateUrl) updateUrl();
}

function initializeSceneSelectors() {
    const scenes = state.scenesIndex?.scenes || [];
    const options = scenes.map((scene) => (
        `<option value="${escapeAttr(scene.id)}">${escapeHtml(scene.label || scene.id)}</option>`
    )).join('');
    document.getElementById('scene-breakdown-selector').innerHTML = options;
    document.getElementById('artifact-scene-selector').innerHTML = options;
}

function initializeArtifactMethodSelector() {
    const selector = document.getElementById('artifact-method-selector');
    const methods = currentArtifactMethods();
    if (methods.length === 0) {
        selector.innerHTML = '<option value="">No methods</option>';
        selector.value = '';
        return;
    }

    selector.innerHTML = methods.map((method) => (
        `<option value="${escapeAttr(method)}">${escapeHtml(method)}</option>`
    )).join('');
    selector.value = state.artifactMethod || methods[0];
}

function renderDatasetOverview() {
    const container = document.getElementById('dataset-overview');
    const metrics = state.overview?.metrics || {};
    const methods = collectMethods(metrics);
    container.innerHTML = `
        ${renderDatasetActions()}
        ${renderMethodFilter('dataset', methods)}
        ${renderCategoryTables(metrics, {
            scope: 'dataset',
            sources: state.overview?.sources || {},
        }) || emptyHtml('No dataset-level eval summaries found.')}
    `;
    bindFilterControls(container);
    bindDatasetActions(container);
}

function renderDatasetActions() {
    return `
        <div class="overview-actions">
            <button type="button" data-action="download-dataset-csv">Download CSV</button>
        </div>
    `;
}

function bindDatasetActions(container) {
    container.querySelector('[data-action="download-dataset-csv"]')?.addEventListener('click', downloadDatasetCsv);
}

function downloadDatasetCsv() {
    const csv = buildDatasetCsv();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `report_${safeFileName(state.dataset?.id || 'dataset')}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

function buildDatasetCsv() {
    const metricsByCategory = state.overview?.metrics || {};
    const methods = selectedMethodsForScope('dataset', metricsByCategory);
    const columns = selectedDatasetMetricColumns(metricsByCategory);
    const rows = [
        ['method', ...columns.map((column) => `${categoryLabel(column.category)}: ${metricLabel(column.category, column.metric)}`)],
    ];

    methods.forEach((method) => {
        rows.push([
            method,
            ...columns.map((column) => csvCellValue(column.normalized[method], column.metric)),
        ]);
    });

    return rows.map((row) => row.map(csvEscape).join(',')).join('\n');
}

function selectedMethodsForScope(scope, metricsByCategory) {
    const methods = collectMethods(metricsByCategory);
    const filter = state.methodFilters.get(scope) || ensureMethodFilter(scope, methods);
    return methods.filter((method) => filter.methods.has(method));
}

function selectedDatasetMetricColumns(metricsByCategory) {
    return sortCategories(Object.keys(metricsByCategory || {})).flatMap((category) => {
        const normalized = normalizeMethodMetrics(metricsByCategory[category]);
        const metricKeys = collectMetricKeys(normalized);
        const tableKey = `dataset:${category}`;
        const filter = state.tableFilters.get(tableKey) || ensureTableFilter(tableKey, metricKeys);
        return metricKeys
            .filter((metric) => filter.metrics.has(metric))
            .map((metric) => ({ category, metric, normalized }));
    });
}

function csvCellValue(metrics, metric) {
    if (!metrics) return '';
    const value = flattenMetrics(metrics)[metric];
    return value === undefined || value === null ? '' : value;
}

function csvEscape(value) {
    const text = String(value);
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function safeFileName(value) {
    return String(value).replace(/[^A-Za-z0-9._-]+/g, '_');
}

function renderSceneSummary() {
    const container = document.getElementById('scene-summary');
    const metrics = state.sceneSummary?.metrics || {};
    const methods = collectMethods(metrics);
    container.innerHTML = `
        ${renderMethodFilter('scene', methods)}
        ${renderCategoryTables(metrics, {
            scope: 'scene',
            sources: state.sceneSummary?.sources || {},
        }) || emptyHtml('No scene-level eval summaries found.')}
    `;
    bindFilterControls(container);
}

async function renderArtifactDetails() {
    const container = document.getElementById('artifact-details');
    const method = state.artifactMethod;
    if (!method) {
        container.innerHTML = emptyHtml('No method selected.');
        return;
    }

    const entry = state.artifactSceneSummary?.methods?.[method];
    if (!entry?.detail_path) {
        container.innerHTML = emptyHtml('No detail JSON found for this method.');
        return;
    }

    const detail = await loadMethodDetail(entry.detail_path);
    container.innerHTML = renderArtifactViewer(detail);
}

async function loadMethodDetail(path) {
    if (!state.detailCache.has(path)) {
        state.detailCache.set(path, fetchJson(path));
    }
    return state.detailCache.get(path);
}

function renderCategoryTables(metricsByCategory, options = {}) {
    const categories = sortCategories(Object.keys(metricsByCategory || {}));
    return categories.map((category) => {
        const payload = metricsByCategory[category];
        const table = renderMetricsTable(category, payload, options);
        if (!table) return '';
        return `
            <section class="category-section">
                <div class="category-title">
                    <div>
                        <h3>${escapeHtml(categoryLabel(category))}</h3>
                        ${categoryHint(category)}
                    </div>
                    ${sourceLink(category, options.sources || {})}
                </div>
                ${table}
            </section>
        `;
    }).join('');
}

function renderMetricsTable(category, payload, options = {}) {
    const normalized = normalizeMethodMetrics(payload);
    const methods = Object.keys(normalized).sort();
    if (methods.length === 0) return '';

    const metricKeys = collectMetricKeys(normalized);
    if (metricKeys.length === 0) return '';

    const tableKey = `${options.scope || 'table'}:${category}`;
    const metricFilter = ensureTableFilter(tableKey, metricKeys);
    const methodFilter = state.methodFilters.get(options.scope || 'table') || ensureMethodFilter(options.scope || 'table', methods);
    const selectedMethods = methods.filter((method) => methodFilter.methods.has(method));
    const selectedMetrics = metricKeys.filter((metric) => metricFilter.metrics.has(metric));

    return `
        ${renderMetricFilter(tableKey, category, metricKeys, metricFilter)}
        ${renderTableBody(category, normalized, selectedMethods, selectedMetrics)}
    `;
}

function renderMethodFilter(scope, methods) {
    if (methods.length === 0) return '';
    const filter = ensureMethodFilter(scope, methods);
    return `
        <div class="section-filter">
            <h3>Methods</h3>
            <div class="table-controls" data-filter-id="${escapeAttr(scope)}:methods">
                <details ${state.openFilters.has(`${scope}:methods`) ? 'open' : ''}>
                    <summary>Selected (${filter.methods.size}/${methods.length})</summary>
                    <div class="filter-actions">
                        <button type="button" data-filter-action="all" data-filter-kind="methods" data-scope="${escapeAttr(scope)}">All</button>
                        <button type="button" data-filter-action="none" data-filter-kind="methods" data-scope="${escapeAttr(scope)}">None</button>
                    </div>
                    <div class="checkbox-grid compact">
                        ${methods.map((method) => renderFilterCheckbox(scope, 'methods', method, filter.methods.has(method))).join('')}
                    </div>
                </details>
            </div>
        </div>
    `;
}

function renderMetricFilter(tableKey, category, metrics, filter) {
    return `
        <div class="table-controls" data-filter-id="${escapeAttr(tableKey)}:metrics">
            <details ${state.openFilters.has(`${tableKey}:metrics`) ? 'open' : ''}>
                <summary>Metrics (${filter.metrics.size}/${metrics.length})</summary>
                <div class="filter-actions">
                    <button type="button" data-filter-action="all" data-filter-kind="metrics" data-table="${escapeAttr(tableKey)}">All</button>
                    <button type="button" data-filter-action="none" data-filter-kind="metrics" data-table="${escapeAttr(tableKey)}">None</button>
                </div>
                <div class="checkbox-grid compact">
                    ${metrics.map((metric) => renderFilterCheckbox(tableKey, 'metrics', metric, filter.metrics.has(metric), metricLabel(category, metric))).join('')}
                </div>
            </details>
        </div>
    `;
}

function renderFilterCheckbox(owner, kind, value, checked, label = value) {
    const ownerAttr = kind === 'methods' ? 'data-scope' : 'data-table';
    return `
        <label class="checkbox-pill">
            <input type="checkbox" data-filter-kind="${escapeAttr(kind)}" ${ownerAttr}="${escapeAttr(owner)}" value="${escapeAttr(value)}" ${checked ? 'checked' : ''}>
            <span>${escapeHtml(label)}</span>
        </label>
    `;
}

function renderTableBody(category, normalized, methods, metrics) {
    if (methods.length === 0 || metrics.length === 0) {
        return emptyHtml('Select at least one method and one metric.');
    }

    const header = `
        <tr>
            <th class="sticky-col">Method</th>
            ${metrics.map((metric) => `<th>${escapeHtml(metricLabel(category, metric))}</th>`).join('')}
        </tr>
    `;
    const ranks = computeMetricRanks(category, normalized, methods, metrics);
    const rows = methods.map((method) => {
        const flat = flattenMetrics(normalized[method]);
        const cells = metrics.map((metric) => {
            const value = flat[metric];
            const rankClass = ranks.get(rankKey(method, metric)) || '';
            return `<td class="${rankClass}">${value === undefined ? '<span class="muted">N/A</span>' : escapeHtml(formatValue(value))}</td>`;
        }).join('');
        return `<tr><th class="sticky-col row-header">${escapeHtml(method)}</th>${cells}</tr>`;
    }).join('');

    return `
        <div class="table-wrap">
            <table>
                <thead>${header}</thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

function computeMetricRanks(category, normalized, methods, metrics) {
    const ranks = new Map();
    metrics.forEach((metric) => {
        if (isStatMetric(metric)) return;
        const values = methods.map((method) => ({
            method,
            value: flattenMetrics(normalized[method])[metric],
        })).filter((entry) => typeof entry.value === 'number' && Number.isFinite(entry.value));
        if (values.length === 0) return;

        const direction = metricDirection(category, metric);
        values.sort((a, b) => direction * (b.value - a.value));

        let rank = 0;
        let previousValue;
        values.forEach((entry, index) => {
            if (previousValue === undefined || entry.value !== previousValue) {
                rank = index + 1;
                previousValue = entry.value;
            }
            if (rank <= 3) {
                ranks.set(rankKey(entry.method, metric), `rank-${rank}`);
            }
        });
    });
    return ranks;
}

function rankKey(method, metric) {
    return `${method}::${metric}`;
}

function metricDirection(category, metric) {
    const normalized = metric.toLowerCase();
    if (category.startsWith('auc') || normalized.startsWith('auc') || normalized.startsWith('racc') || normalized.startsWith('tacc')) {
        return 1;
    }
    if (normalized.includes('accuracy') || normalized.includes('completion') || normalized.includes('precision') || normalized.includes('recall') || normalized.includes('fscore')) {
        return 1;
    }
    return -1;
}

function bindFilterControls(container) {
    container.querySelectorAll('details').forEach((details) => {
        details.addEventListener('toggle', (event) => {
            const filterId = event.target.parentElement?.dataset?.filterId;
            if (!filterId) return;
            if (event.target.open) {
                state.openFilters.add(filterId);
            } else {
                state.openFilters.delete(filterId);
            }
        });
    });

    container.querySelectorAll('input[data-filter-kind]').forEach((checkbox) => {
        checkbox.addEventListener('change', (event) => {
            const kind = event.target.dataset.filterKind;
            if (kind === 'methods') {
                const scope = event.target.dataset.scope;
                const filter = state.methodFilters.get(scope);
                if (!filter) return;
                updateSet(filter.methods, event.target.value, event.target.checked);
                rerenderScope(scope);
                return;
            }

            const tableKey = event.target.dataset.table;
            const filter = state.tableFilters.get(tableKey);
            if (!filter) return;
            updateSet(filter.metrics, event.target.value, event.target.checked);
            rerenderTableScope(tableKey);
        });
    });

    container.querySelectorAll('button[data-filter-action]').forEach((button) => {
        button.addEventListener('click', (event) => {
            const kind = event.target.dataset.filterKind;
            const action = event.target.dataset.filterAction;
            if (kind === 'methods') {
                const scope = event.target.dataset.scope;
                const filter = state.methodFilters.get(scope);
                if (!filter) return;
                filter.methods = action === 'all' ? new Set(filter.methodsAll) : new Set();
                rerenderScope(scope);
                return;
            }

            const tableKey = event.target.dataset.table;
            const filter = state.tableFilters.get(tableKey);
            if (!filter) return;
            filter.metrics = action === 'all' ? new Set(filter.metricsAll) : new Set();
            rerenderTableScope(tableKey);
        });
    });
}

function updateSet(set, value, enabled) {
    if (enabled) {
        set.add(value);
    } else {
        set.delete(value);
    }
}

function rerenderScope(scope) {
    if (scope === 'dataset') {
        renderDatasetOverview();
    } else if (scope === 'scene') {
        renderSceneSummary();
    }
}

function rerenderTableScope(tableKey) {
    if (tableKey.startsWith('dataset:')) {
        renderDatasetOverview();
    } else if (tableKey.startsWith('scene:')) {
        renderSceneSummary();
    }
}

function ensureTableFilter(tableKey, metrics) {
    const existing = state.tableFilters.get(tableKey);
    if (!existing) {
        const filter = {
            metrics: new Set(metrics),
            metricsAll: metrics,
        };
        state.tableFilters.set(tableKey, filter);
        return filter;
    }

    existing.metricsAll = metrics;
    existing.metrics = new Set([...existing.metrics].filter((metric) => metrics.includes(metric)));
    return existing;
}

function ensureMethodFilter(scope, methods) {
    const existing = state.methodFilters.get(scope);
    if (!existing) {
        const filter = {
            methods: new Set(methods),
            methodsAll: methods,
        };
        state.methodFilters.set(scope, filter);
        return filter;
    }

    existing.methodsAll = methods;
    existing.methods = new Set([...existing.methods].filter((method) => methods.includes(method)));
    return existing;
}

function collectMethods(metricsByCategory) {
    const methods = new Set();
    Object.values(metricsByCategory || {}).forEach((payload) => {
        Object.keys(normalizeMethodMetrics(payload)).forEach((method) => methods.add(method));
    });
    return [...methods].sort();
}

function normalizeMethodMetrics(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return {};
    const methodKeys = Object.keys(payload).filter((key) => payload[key] && typeof payload[key] === 'object' && !Array.isArray(payload[key]));
    if (methodKeys.length === 0) {
        return { metrics: payload };
    }

    const result = {};
    methodKeys.forEach((method) => {
        result[method] = payload[method];
    });
    return result;
}

function collectMetricKeys(normalized) {
    const metricKeys = new Set();
    Object.values(normalized).forEach((metrics) => {
        Object.keys(flattenMetrics(metrics)).forEach((key) => metricKeys.add(key));
    });
    return [...metricKeys].sort(metricNameCompare);
}

function renderArtifactViewer(detail) {
    const jsonArtifacts = detail.artifacts?.json || {};
    const imageArtifacts = detail.artifacts?.images || {};
    const jsonLinks = Object.entries(jsonArtifacts).sort().map(([category, path]) => (
        `<a class="artifact-link" href="${escapeAttr(path)}" target="_blank" rel="noreferrer">${escapeHtml(categoryLabel(category))} JSON</a>`
    )).join('');
    const imageSections = sortCategories(Object.keys(imageArtifacts)).map((category) => {
        const images = imageArtifacts[category] || [];
        const gallery = images.length > 0
            ? images.map((path) => `
                <a href="${escapeAttr(path)}" target="_blank" rel="noreferrer">
                    <img src="${escapeAttr(path)}" alt="${escapeAttr(categoryLabel(category))}">
                </a>
            `).join('')
            : '<div class="empty">No image artifacts.</div>';
        return `
            <section class="artifact-category artifact-category-${escapeAttr(category)}">
                <h3>${escapeHtml(categoryLabel(category))}</h3>
                <div class="gallery gallery-${escapeAttr(category)}">${gallery}</div>
            </section>
        `;
    }).join('');

    return `
        <div class="artifact-header">
            <div>
                <h3>${escapeHtml(detail.method)}</h3>
                <p>${escapeHtml(detail.scene)}</p>
            </div>
            <div class="artifact-links">${jsonLinks || '<span class="muted">No JSON artifacts</span>'}</div>
        </div>
        ${imageSections || emptyHtml('No image artifacts found for this method.')}
    `;
}

function sourceLink(category, sources) {
    const path = sources[category];
    if (!path) return '';
    return `<a class="source-link" href="${escapeAttr(path)}" target="_blank" rel="noreferrer">source</a>`;
}

function flattenMetrics(value, prefix = '') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    const result = {};
    Object.entries(value).forEach(([key, child]) => {
        const nextKey = prefix ? `${prefix}.${key}` : key;
        if (isDisplayValue(child)) {
            result[nextKey] = child;
        } else if (child && typeof child === 'object' && !Array.isArray(child)) {
            Object.assign(result, flattenMetrics(child, nextKey));
        }
    });
    return result;
}

function isDisplayValue(value) {
    return typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean' || value === null;
}

function metricLabel(category, key) {
    const definitions = state.manifest?.metric_definitions || {};
    const fullKey = `${category}.${key}`;
    const definitionLabel = definitions[fullKey]?.label
        || (category.startsWith('auc_') ? definitions[`auc.${key}`]?.label : null);
    return normalizeMetricLabel(definitionLabel || key);
}

function normalizeMetricLabel(label) {
    const parts = String(label).split('.');
    return parts.map(normalizeMetricLabelPart).join(' / ');
}

function normalizeMetricLabelPart(part) {
    const compact = part.trim().replace(/\s+/g, ' ');
    const aucMatch = compact.match(/^(AUC|Racc|Tacc)[\s_@-]*(\d+)$/i);
    if (aucMatch) {
        return `${aucMatch[1].toUpperCase()}@${Number(aucMatch[2])}`;
    }

    return compact
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (letter) => letter.toUpperCase())
        .replace(/\b(Auc|Racc|Tacc)\s+(\d+)\b/g, (_, name, threshold) => `${name.toUpperCase()}@${Number(threshold)}`)
        .replace(/\b(Auc|Racc|Tacc)@(\d+)\b/g, (_, name, threshold) => `${name.toUpperCase()}@${Number(threshold)}`);
}

function categoryHint(category) {
    if (category === 'auc_micro') {
        return '<p class="category-hint">Pair-level aggregation across all evaluated frame pairs.</p>';
    }
    if (category === 'auc_macro') {
        return '<p class="category-hint">Scene-level aggregation: average of per-scene scores.</p>';
    }
    return '';
}

function categoryLabel(category) {
    return CATEGORY_LABELS[category] || prettifyKey(category);
}

function sortCategories(categories) {
    return [...categories].sort((a, b) => {
        const ai = CATEGORY_ORDER.indexOf(a);
        const bi = CATEGORY_ORDER.indexOf(b);
        if (ai === -1 && bi === -1) return a.localeCompare(b);
        if (ai === -1) return 1;
        if (bi === -1) return -1;
        return ai - bi;
    });
}

function metricNameCompare(a, b) {
    const rankDiff = metricRank(a) - metricRank(b);
    if (rankDiff !== 0) return rankDiff;
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

function metricRank(key) {
    return isStatMetric(key) ? 1 : 0;
}

function isStatMetric(key) {
    const normalized = key.toLowerCase();
    return normalized.startsWith('num_')
        || normalized.includes('.num_')
        || normalized.includes('count')
        || normalized.includes('pairs')
        || normalized.includes('frames')
        || normalized.includes('scenes');
}

function currentArtifactMethods() {
    return [...(state.artifactScene?.methods || [])].sort();
}

function findDataset(datasetId) {
    return state.manifest?.datasets?.find((dataset) => dataset.id === datasetId) || null;
}

function findScene(sceneId) {
    return state.scenesIndex?.scenes?.find((scene) => scene.id === sceneId) || null;
}

async function fetchJson(path) {
    const response = await fetch(path);
    if (!response.ok) {
        throw new Error(`${path}: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

function updateUrl() {
    if (!state.dataset) return;
    const url = new URL(window.location.href);
    url.searchParams.set('dataset', state.dataset.id);
    if (state.scene) url.searchParams.set('scene', state.scene.id);
    if (state.artifactScene) url.searchParams.set('artifact_scene', state.artifactScene.id);
    if (state.artifactMethod) url.searchParams.set('artifact_method', state.artifactMethod);
    url.searchParams.delete('method');
    url.searchParams.delete('methods');
    window.history.replaceState({}, '', url);
}

function renderEmpty(elementId, message) {
    document.getElementById(elementId).innerHTML = emptyHtml(message);
}

function emptyHtml(message) {
    return `<div class="empty">${escapeHtml(message)}</div>`;
}

function setStatus(message, type = 'ok') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${type}`;
}

function formatValue(value) {
    if (value === null) return 'null';
    if (typeof value === 'number') {
        if (!Number.isFinite(value)) return String(value);
        const abs = Math.abs(value);
        if (abs !== 0 && (abs < 0.0001 || abs >= 100000)) return value.toExponential(4);
        return Number(value.toPrecision(6)).toString();
    }
    return String(value);
}

function formatDate(value) {
    if (!value) return 'unknown time';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function prettifyKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function escapeAttr(value) {
    return escapeHtml(value);
}
