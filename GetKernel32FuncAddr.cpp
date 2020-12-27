//  GetKernel32FuncAddr.cpp : 
#include <stdio.h>
#include <stdlib.h>

unsigned long GetKernel32FuncAddr()
{
    unsigned long pBaseOfModule, pNameOfModule;
    unsigned long pAddressOfFunctions, pAddress0fNames;

    __asm{
        mov edx, fs:30h         ; PEB base
        mov	edx, [edx+0ch]      ; PEB_LER_DATA
        // base of ntdll.dll=====================
        mov edx, [edx+1ch]      ; The first element of InInitOrderModuleList
        // base of kernel32.dll=====================
        mov edx,[edx]           ; Next element
        mov eax, [edx+8]        ; Base address of second module
        mov pBaseOfModule,eax	; Save it to local variable
        mov ebx, eax            ; Base address of kernel32.dll, save it to ebx
        // get the addrs of first function =========
        mov edx,[ebx+3ch]       ; e_lfanew
        mov edx,[edx+ebx+78h]   ; DataDirectory[0]
        add edx,ebx             ; RVA + base
        mov esi,edx             ; Save first DataDirectory to esi
        // get fields of IMAGE_EXPORT_DIRECTORY pNameOfModule
        mov edx,[esi+0ch]           ; Name
        add edx,ebx                 ; RVA + base
        mov pNameOfModule,edx       ; Save it to local variable
        mov edx,[esi+1ch]           ; AddressOfFunctions RVA
        add edx,ebx                 ; RVA + base
        mov pAddressOfFunctions,edx ; Save it to local variable
        mov edx,[esi+20h]       ; AddressOfNames RVA
        add edx,ebx             ; RVA + base
        mov pAddress0fNames,edx ; Save it to local variable
    }
    printf("Name of Module:%s\n\tBase of Moudle=%p\n",
            (char *)pNameOfModule,pBaseOfModule);
    printf("First Function:\n\tAddress=0x%p\n\tName=%s\n",
        (pBaseOfModule + *((unsigned long *) (pAddressOfFunctions))),
        (char *)(pBaseOfModule + *((unsigned long *) (pAddress0fNames)))) ;
}

unsigned long GetDLLAddr(int nIndex)
{
    unsigned long pBaseOfModule=0, pNameOfModule;
    unsigned long pAddressOfFunctions, pAddress0fNames;

    __asm{
        mov ecx, nIndex         ; 
        mov edx, fs:30h         ; PEB base
        mov	edx, [edx+0ch]      ; PEB_LER_DATA
        // base of ntdll.dll=====================
        mov edx, [edx+1ch]      ; The first element of InInitOrderModuleList
        cmp ecx,0
        je  SaveBaseAddr;
        // base of Next element=====================
NextElement:
        mov edx,[edx]           ; Next element
        loop NextElement        ;
SaveBaseAddr:        
        mov eax, [edx+8]        ; Base address of second module
        mov pBaseOfModule,eax	; Save it to local variable
        mov ebx, eax            ; Base address of kernel32.dll, save it to ebx
        cmp eax,0
        je  EndOfGetDLLAddr;
        // get the addrs of first function =========
        mov edx,[ebx+3ch]       ; e_lfanew
        mov edx,[edx+ebx+78h]   ; DataDirectory[0]
        add edx,ebx             ; RVA + base
        mov esi,edx             ; Save first DataDirectory to esi
        // get fields of IMAGE_EXPORT_DIRECTORY pNameOfModule
        mov edx,[esi+0ch]           ; Name
        add edx,ebx                 ; RVA + base
        mov pNameOfModule,edx       ; Save it to local variable
        mov edx,[esi+1ch]           ; AddressOfFunctions RVA
        add edx,ebx                 ; RVA + base
        mov pAddressOfFunctions,edx ; Save it to local variable
        mov edx,[esi+20h]       ; AddressOfNames RVA
        add edx,ebx             ; RVA + base
        mov pAddress0fNames,edx ; Save it to local variable
EndOfGetDLLAddr:                ;
    }

    printf("Base of the Moudle[%d]=%p\n",nIndex, pBaseOfModule);

    if(pBaseOfModule !=0)
    {
        printf("Name of Module: %s\n", (char *)pNameOfModule);
        printf("First Function: \n\tAddress=0x%p\n\tName=%s\n\n",
            (pBaseOfModule + *((unsigned long *) (pAddressOfFunctions))),
            (char *)(pBaseOfModule + *((unsigned long *) (pAddress0fNames)))) ;
    }else{ puts("");}

    return pBaseOfModule;
}

void main(void)
{
    GetKernel32FuncAddr();
    for(int nIndex=0; nIndex<4; nIndex++)
    {
        //printf("\nDo GetDLLAddr(%d)\n", nIndex);
        GetDLLAddr(nIndex);
    }
}