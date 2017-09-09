# Input and output

Input and output could be performed in three different modes:

- Images (`.bmp`)
- Binary files (`.dat`)
- Plain text files (`.txt`)

For all details see `--help`.

# Output

To save results use `--save-res` command line argument, to save materials use `--save-materials`.

## Images mode

In addition, to perform save to `.bmp` images add `--save-as-bmp` command line argument.

File names are setup in the next manner:
`<Type time step>[<Time step>]_rank-<Process Id>_<Field name>-<Type>.bmp`

- Type time step could be one of three: `current`, `previous`, `previous2`.
- Time step is the number of time step, after which results are saved, except for materials, for which time step is always 0.
- Process id is the id of the process, performing saving. Currently, only the process with 0 id is performing saving.
- Field name is the special string for mode plus the name of the field, e.g. `Ex`, `Ey`, `Ez` for grid name, and `3D-in-time` string for 3D mode, `2D-TMz-in-time` and `2D-TEz-in-time` for 2D mode. In case of 3D mode images are saved for each layer orthogonal to specified axis, so name is `Ex40` for `Ex` grid and 40 layer.
- Type is one of three: `Re` for real part, `Im` for imaginary part, `Mod` for module.

Example for 2D mode: `current[300]_rank-0_2D-TMz-in-time-Ez-Mod.bmp`.
Example for 3D mode: `current[300]_rank-0_3D-in-time-Ez40-Mod.bmp`.

Also, for each .bmp image additional .txt file is saved with maximum and minimum values.

### Additional parameters

- `--palette-gray` or `--palette-rgb` for different color schemes
- [WIP] `--orth-axis-x`, or `--orth-axis-y`, or `--orth-axis-z` for orthogonal axis. (default is `--orth-axis-z`)
- [WIP] start and end coordinates to save or load

## Binary mode

In addition, to perform save to `.dat` binary files add `--save-as-dat` command line argument.

File names are similar to .bmp, except that there is no `Type` and field name doesn't have index of layer.

Example for 3D mode: `current[300]_rank-0_3D-in-time-Ez.dat`.

### Format

This file is binary and grid values are saved as is.

## Plain text mode

In addition, to perform save to `.txt` plain text files add `--save-as-txt` command line argument.

File names are same as for .dat.

Example for 3D mode: `current[300]_rank-0_3D-in-time-Ez.txt`.

### Format
Single line in file has the next format

```
xcoord ycoord zcoord re-value im-value
```

where x,y,z coords are coordinates of the point in grid, value is the value of the component in that point. im-value is absent in non-complex value builds, y and z coords are absent for 1D mode, z coord is missing for 2D mode.

# Input

For all three types of file load is performed in the same manner. Be sure to load files with the same size as the grid.

For example, to load `eps` use `--load-eps-from-file <filename>` command line argument. Type of file is determined by file extension, i.e. the string after the last dot in filename. In case of `.txt` and `.dat` the file is simply loaded. However, `.bmp` mode is the exception here, because in case of 3D mode, there are multiple files. So, the `<filename>` specified is added additional `<Layer index>-<Type>.bmp`, e.g. if filename is `current[300]_rank-0_3D-in-time-Ez.bmp`, then one of the loaded files is `current[300]_rank-0_3D-in-time-Ez.bmp40-Re.bmp`.
