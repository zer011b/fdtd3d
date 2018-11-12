# Input and output

Input and output could be performed in three different modes:

- Images (`.bmp`)
- Binary files (`.dat`)
- Plain text files (`.txt`)

For all details see `--help`.

# Output

To save results use `--save-res` command line argument, to save materials use `--save-materials`.

To perform save to `.bmp` images add `--save-as-bmp` command line argument.
To perform save to `.dat` binary files add `--save-as-dat` command line argument.
To perform save to `.txt` plain text files add `--save-as-txt` command line argument.

File names for `.txt` and `.dat` are setup in the next manner:
```
previous-<Time step offset>_[timestep=<Time step>]_[pid=<Process Id>]_[name=<Field name>].{txt,dat,bmp}
```

- Time step offset indicates offset of time step from current (i.e. current time step has offset 0, previous time step has offset 1, etc.).
- Time step is the number of time step, after which grids are saved.
- Process id is the id of the process, performing saving.
- Field name is the name of the field, e.g. `Ex`, `Ey`, `Ez`.

File names for `.bmp` are setup in the next manner:
```
previous-<Time step offset>_[timestep=<Time step>]_[pid=<Process Id>]_[name=<Field name>]_[<Type>]<Additional string>.{txt,dat,bmp}
```

- Type is one of three: `Re` for real part, `Im` for imaginary part, `Mod` for module.
- In case of 3D mode images are saved for each layer orthogonal to specified axis. Name has to consider coordinate on this axis, thus, additional string from above becomes `_[coord=<coord val>]`.

Example for 2D mode `.txt`: `previous-0[timestep=300]_[pid=0]_[name=Ez].txt`.

Example for 3D mode `.dat`: `previous-0[timestep=300]_[pid=0]_[name=Ez].dat`.

Example for 2D mode `.bmp`: `previous-0[timestep=300]_[pid=0]_[name=Ez]_[mod].bmp`.

Example for 3D mode `.bmp`: `previous-0[timestep=300]_[pid=0]_[name=Ez]_[mod]_[coord=40].bmp`.

## Images mode

For each .bmp image additional .txt file is saved with maximum and minimum values.

Additional parameters:
- `--palette-gray` or `--palette-rgb` for different color schemes
- `--orth-axis-x`, or `--orth-axis-y`, or `--orth-axis-z` for orthogonal axis. (default is `--orth-axis-z`)
- `--save-start-coord-x`, etc., for start and end coordinates to save or load

## Binary mode

This file is binary and grid values are saved as is.

## Plain text mode

Single line in file has the next format

```
xcoord ycoord zcoord re-value im-value
```

where x,y,z coords are coordinates of the point in grid, value is the value of the component in that point. im-value is absent in non-complex value builds, y and z coords are absent for 1D mode, z coord is missing for 2D mode.

# Input

For all three types of file load is performed in the same manner.

** Be sure to load files with the same size as the grid. Assert will fire if this is not correct. **

For example, to load `eps` use `--load-eps-from-file <filename>` command line argument. Type of file is determined by file extension, i.e. the string after the last dot in filename. In case of `.txt` and `.dat` the file is simply loaded.

However, `.bmp` mode is the exception here, because there are multiple files for real/imag values and in 3D mode there are different coordinates on orthogonal axis. Latter case is not supported currently, as it is not considered quite useful (i.e. drawing multiple grids by hand in image editor for 3D mode doesn't seem to be common approach).

To handle former, the `<filename>` specified is added additional `_[real].bmp`, `_[imag].bmp`, e.g. if filename passed as argument is `previous-0[timestep=300]_[pid=0]_[name=Ez].bmp`, then one of the loaded files is `previous-0[timestep=300]_[pid=0]_[name=Ez]_[real].bmp`.
