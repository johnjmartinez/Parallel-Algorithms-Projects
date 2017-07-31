/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef TIMER_H
#define TIMER_H

#include <stdlib.h>
#include <sys/time.h>

struct timeval timerStart;

void StartTimer()
{
   gettimeofday(&timerStart, NULL);
}

// time elapsed in ms
double GetTimer()
{
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}
#endif // TIMER_H
