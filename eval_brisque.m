function eval_brisque(varargin)
    p = inputParser;
    addOptional(p, 'FolderPath', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    
    folder_path = p.Results.FolderPath;
    
    if ~isempty(folder_path)
        disp(['Folder path received: ', folder_path]);

        files = dir(fullfile(folder_path, '*.png')); 
        images = cell(1, numel(files));

        for i = 1:numel(files)
            image_path = fullfile(folder_path, files(i).name);
            images{i} = imread(image_path);
        end

        fprintf('Number of images: %d.\n', numel(images));

        brisque_list = zeros(1,numel(images));
        for i = 1:numel(images)
            brisque_list(i) = brisque(images{i});
        end

        brisque_avg = mean(brisque_list);
        brisque_avg
    else
        disp('No folder path provided.');
    end
end