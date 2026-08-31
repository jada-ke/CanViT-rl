% Batch precompute ADE20K saliency maps with MATLAB reference toolboxes.
%
% Examples from the repository root:
%   matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency"
%   matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','gbvs','toolbox_root','/path/to/gbvs')"
%   matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','itti','toolbox_root','/path/to/SaliencyToolbox')"
%
% Outputs one .mat per ADE image stem. Each file contains `salmap`, which the
% Python converter reads directly:
%   uv run python scripts/saliency/precompute_saliency_maps.py \
%       --method itti \
%       --external-map-dir results/matlab_itti_maps

function run_ade20k_saliency(varargin)
    parser = inputParser;
    addParameter(parser, 'method', 'itti', @ischar);
    addParameter(parser, 'toolbox_root', '', @ischar);
    addParameter(parser, 'dataset', 'datasets/ADE20k', @ischar);
    addParameter(parser, 'split', 'validation', @ischar);
    addParameter(parser, 'output_dir', '', @ischar);
    addParameter(parser, 'preview_samples', 8, @isnumeric);
    addParameter(parser, 'preview_size', 512, @isnumeric);
    parse(parser, varargin{:});

    method = char(parser.Results.method);
    toolbox_root = char(parser.Results.toolbox_root);
    dataset_root = char(parser.Results.dataset);
    split = char(parser.Results.split);
    output_dir = char(parser.Results.output_dir);
    preview_samples = parser.Results.preview_samples;
    preview_size = parser.Results.preview_size;

    repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    if ~isfolder(fullfile(repo_root, 'scripts'))
        repo_root = pwd;
    end

    if isempty(toolbox_root) && strcmpi(method, 'gbvs')
        % Problem: the preferred GBVS install is a sibling of this repo.
        % Solution: default GBVS to that sibling folder when no toolbox_root is
        % passed. Result: the batch command can stay short while still allowing
        % explicit overrides for alternate installs.
        toolbox_root = fullfile(repo_root, '..', 'gbvs');
    end

    if ~isempty(toolbox_root)
        addpath(genpath(toolbox_root));
    end

    if ~isfolder(dataset_root)
        dataset_root = fullfile(repo_root, dataset_root);
    end
    image_dir = fullfile(dataset_root, 'images', split);
    if ~isfolder(image_dir)
        error('Image directory not found: %s', image_dir);
    end

    if isempty(output_dir)
        output_dir = fullfile(repo_root, 'results', ['matlab_' method '_maps']);
    elseif ~isfolder(fileparts(output_dir)) && ~startsWith(output_dir, filesep)
        output_dir = fullfile(repo_root, output_dir);
    end
    preview_dir = fullfile(output_dir, 'previews');
    if ~isfolder(output_dir), mkdir(output_dir); end
    if ~isfolder(preview_dir), mkdir(preview_dir); end

    images = [dir(fullfile(image_dir, '*.jpg')); dir(fullfile(image_dir, '*.png')); dir(fullfile(image_dir, '*.jpeg'))];
    if isempty(images)
        error('No images found in %s', image_dir);
    end

    fprintf('Running %s saliency on %d %s images\n', method, numel(images), split);
    for i = 1:numel(images)
        image_path = fullfile(images(i).folder, images(i).name);
        [~, stem, ~] = fileparts(image_path);
        img = imread(image_path);

        salmap = compute_saliency_map(img, method);
        save(fullfile(output_dir, [stem '.mat']), 'salmap');
        if i <= preview_samples
            % Problem: SaliencyToolbox can return tiny pyramid-resolution maps,
            % making raw previews hard to inspect. Solution: resize only the
            % preview image to a fixed square while preserving raw .mat values.
            % Result: previews are readable and eval conversion remains numeric.
            preview = imresize(mat2gray(salmap), [preview_size preview_size], 'bilinear');
            imwrite(preview, fullfile(preview_dir, [stem '.png']));
        end

        if mod(i, 25) == 0 || i == numel(images)
            fprintf('Saved %s (%d/%d)\n', stem, i, numel(images));
        end
    end
    fprintf('Saved MATLAB saliency maps to %s\n', output_dir);
end

function salmap = compute_saliency_map(img, method)
    switch lower(method)
        case 'itti'
            salmap = call_saliency_toolbox(img);
            return
        case 'gbvs'
            assert_exist('gbvs');
            % Problem: partial GBVS installs can expose gbvs.m while missing
            % required helper folders. Solution: check one core utility before
            % launching the full ADE20K batch. Result: path/install mistakes fail
            % early with the missing dependency name instead of after work starts.
            assert_exist('mySubsample');
            result = gbvs(img);
        case 'aws'
            result = call_aws(img);
        otherwise
            error('Unsupported method: %s', method);
    end
    salmap = extract_saliency_map(result);
end

function salmap = call_saliency_toolbox(img)
    assert_exist('initializeImage');
    assert_exist('defaultSaliencyParams');
    assert_exist('makeSaliencyMap');
    img_struct = initializeImage(img);
    params = defaultSaliencyParams(img_struct.size, 'dyadic');
    if img_struct.dims == 2
        assert_exist('removeColorFeatures');
        params = removeColorFeatures(params);
    end
    % Problem: multiple Itti labels made baseline plots ambiguous. Solution:
    % reserve `method=itti` for the SaliencyToolbox Walther/Koch pipeline.
    % Result: Python cache/eval outputs have one canonical Itti method name.
    salmap_struct = makeSaliencyMap(img_struct, params);
    if isstruct(salmap_struct) && isfield(salmap_struct, 'data')
        salmap = salmap_struct.data;
    else
        salmap = extract_saliency_map(salmap_struct);
    end
end

function salmap = call_aws(img)
    % Problem: AWS distributions use different entry-point names across
    % releases. Solution: try the common names and fail with a clear message.
    % Result: users can point toolbox_root at their AWS checkout without editing
    % this batch runner when the standard function name is present.
    if exist('aws', 'file') == 2
        salmap = aws(img);
    elseif exist('AWS', 'file') == 2
        salmap = AWS(img);
    elseif exist('awsSaliency', 'file') == 2
        salmap = awsSaliency(img);
    else
        error('Could not find an AWS saliency function on the MATLAB path.');
    end
end

function salmap = extract_saliency_map(result)
    if isnumeric(result)
        salmap = result;
        return
    end
    field_candidates = {'master_map_resized', 'master_map', 'saliencyMap', 'salmap', 'map'};
    for k = 1:numel(field_candidates)
        field = field_candidates{k};
        if isstruct(result) && isfield(result, field)
            salmap = result.(field);
            return
        end
    end
    error('Could not extract saliency map from toolbox result.');
end

function assert_exist(function_name)
    kind = exist(function_name, 'file');
    % Problem: GBVS helpers such as mySubsample are compiled MEX files on
    % modern macOS, and MATLAB reports those as exist(...)=3 instead of the
    % exist(...)=2 used for .m files. Solution: accept both source functions
    % and compiled MEX helpers. Result: successful GBVS compiles are recognized.
    if ~(kind == 2 || kind == 3)
        error('Required saliency function not found on path: %s', function_name);
    end
end
