#ifndef EXECUTE_H
#define EXECUTE_H

typedef double FieldValue;

extern void execute (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy, FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
              int sx, int sy,
              FieldValue gridTimeStep,
              FieldValue gridStep,
              int totalStep);

#endif
